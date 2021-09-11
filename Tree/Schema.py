"""
A Schema represents the possible objects, relationships, and attributes of those
objects and relationships. Distinctions use the schema for determining what things
they can select from when generating a random instance of themselves. If its not
in the schema then no distinction should use it, even if its in the data.


"""
from collections import defaultdict


from .STAttribute import DataTypes


# The loaded in schema is turned into this object which tracks all the object types, relation types,
# and attributes
class Schema(object):
    """The Schema describes the types of objects and relations in.

    :var objects: (Dict[object.id => :class:`~srpt.Schema.SchemaObject`])
        the different types of objects that can exist in a graph
    :var relations: (Dict[relation.id => :class:`~srpt.Schema.SchemaRelation`])
        the different types of relationships that can exist in a graph
    :var attributes: (Dict[attribute.parent.id => Dict[relation.name => :class:`~srpt.Schema.SchemaAttribute`]])
        the different types of attributes that can exist in a graph for each type
        of object and relation indexed by object/relation id
    :var all_attributes: (Dict[(attribute.parent.id, relation.name) => :class:`~srpt.Schema.SchemaAttribute`])
        the attributes in the graph indexed by a tuple of (object/relation id, relation name)
    :var objects_and_relations:
        a list of the objects and relations
    """

    def __init__(self):
        # objects defined in the schema
        self.objects = {}


        # relations defined in the schema
        self.relations = {}
        self.relation_names = []

        # attributes for the objects and relations defined in the schema
        self.attributes = defaultdict(dict)
        self.all_attributes = {}

        self.objects_and_relations = []

        # internally used for unique id tracking
        self.next_id = 1
        self.max_id = 0

        # internally used for reflection building
        self.relations_by_triple = {}

        # schema level attributes
        self.attrs = {}

        # field data objects name->type
        self.fielddata = {}

    def removeIgnored(self):
        """Removes all ignored objects, relations, and attribubutes from the schema
        by checking their *ignore* attribute. This prevents them from being used
        by distinctions when randomly generating them. 
        """

        objects = set()
        for key, item in self.objects.items():
            if item.ignore:
                objects.add(key)

        for key in objects:
            self.objects.pop(key)

        relations = set()
        for key, item in self.relations.items():
            if item.ignore:
                relations.add(key)
        for key in relations:
            self.relations.pop(key)

        attributes = set()
        for key, item in self.all_attributes.items():
            if item.ignore or item.parent.ignore:
                attributes.add(key)
        for key in attributes:
            attrib = self.all_attributes.pop(key)
            self.attributes[key[0]].pop(key[1])
            attrib.parent.attributes.pop(attrib.name)
        for key in self.attributes.keys():
            if len(self.attributes[key]) == 0:
                self.attributes.pop(key)

        self.relation_names = [rel.name for rel in self.relations.values()]
        self.objects_and_relations = [item for item in self.objects_and_relations if
                                      item.type not in objects and item.type not in relations]

        if False:
            import sys
            from pprint import pprint
            #pprint(self.objects)
            #pprint(self.relations)
            #pprint(dict(self.attributes))
            #pprint(self.objects_and_relations)
            sys.exit()


class SchemaGraph(object):
    """The schema object representing a graph.

    :var schema: the schema this graph belongs to
    """

    def __init__(self, schema, attrs):
        """
        :param schema: the schema this graph belongs too
        :param attrs: this is *ignored*
        """

        self.attributes = {}
        self.schema = schema
        # attache the schema to this graph
        self.schema.graph = self
        self.type = 'graph'
        self.id = 'graph'

    def __str__(self):
        s = 'Graph['
        s = '%s\n\t attribs=[%s]' % (s, '\n\t\t  '.join(map(repr, self.attributes.values())))
        s = '%s]' % (s)
        return s

    def __repr__(self):
        return 'Graph[]'


# An object represented in the schema
class SchemaObject(object):
    """
    :var ignore: (``Boolean``) ignore this object?
    :var type: the type/name of this object (same as ``id``)
    :var id: the name/type of this object (same as ``type``)
    :var attributes: (Dict[attribute.name => :class:`~.SchemaAttribute`])
        all the attributes of this object
    :var out_relations: (Dict[relation.id => :class:`~.SchemaRelation`])
        the relations that go out of this object (i.e. this object is their source)
    :var in_relations: (Dict[relation.id => :class:`~.SchemaRelation`])
        the relations that come into this object (i.e. this object is their target)
    """

    def __init__(self, schema, attrs):
        """Create a SchemaObject which represents a type of object

        :param schema: the schema this type of object belongs too
        :param attrs: the attributes to use in setting up the object 

                * ignore - should this object be ignored
                * type - the type of the object
        """

        self.ignore = attrs.get('ignore', False)
        self.type = attrs.get('type')
        self.attributes = {}
        self.in_relations = {}
        self.out_relations = {}
        self.schema = schema

        self.id = self.type

        schema.objects[self.id] = self
        schema.objects_and_relations.append(self)

    def __str__(self):
        s = 'SchemaObject[type=%s,' % self.type
        s = '%s\n\t attribs=[%s]' % (s, '\n\t\t  '.join(map(repr, self.attributes.values())))
        s = '%s\n\t in-rels=[%s]' % (s, '\n\t\t  '.join(map(repr, self.in_relations.values())))
        s = '%s\n\tout-rels=[%s]' % (s, '\n\t\t  '.join(map(repr, self.out_relations.values())))
        s = '%s]' % (s)
        return s

    def __repr__(self):
        return 'ScObject[type=%s]' % (self.type)


# An attribute represented in the schema
class SchemaAttribute(object):
    """Represents a possible attribute of an object or relation.

    :var ignore: (``Boolean``) should this attribute be ignored?
    :var name: the name of this attribute
    :var parent: the parent object/relation this attribubutes belongs to
    :var id: unique id of this attribute. a tuple of *(parent.id, self.name)*
    :var type: the :data:`~srpt.STAttribute.DataTypes` of the attribute
    """

    def __init__(self, name, parent, attrs, no_schema=False):
        """Creates a SchemaAttribute object. It will automatically be added to
        its parents *attributes* dict, to *schema.attributes*, and to 
        *schema.all_attributes*. The schema is determined by checking:
        *parent.schema*. 

        :param name: the name of the attribute
        :param parent: the object or relation this attribute belongs to
        :type parent: :class:`~.SchemaObject` or :class:`~.SchemaRelation`
        :param attrs: the attributes to use in setting up the object

                * ignore - should this object be ignored
                * type - the datatype of the attribute. one of the enum values from
                  :data:`STAttribute.DataTypes` or a string that is the name of one
                  of the enum values
                * others are set as attributes of this object
        :param no_schema: don't update the schema's *attribues* and *all_attributes*
            dicts
        """

        self.ignore = attrs.get('ignore', False)
        self.__attr_names = []
        for attr, value in attrs.items():
            self.__attr_names.append(attr)
            setattr(self, attr, value)

        self.name = name
        self.parent = parent
        self.parent.attributes[self.name] = self
        self.id = (self.parent.id, self.name)

        if hasattr(parent, 'schema'):
            self.schema = parent.schema
        self.no_schema = no_schema

        if not self.no_schema:
            self.schema.attributes[self.parent.id][self.name] = self
            self.schema.all_attributes[(self.parent.id, self.name)] = self

        if not hasattr(self, 'type'):
            self.type = DataTypes.Discrete  # @UndefinedVariable
        elif type(self.type) is str:
            # handle the XML encoding of DataType names.
            if self.type not in DataTypes:
                parts = self.type.split('-')
                parts = map(str.capitalize, parts)
                self.type = DataTypes.get(''.join(parts))

    def dup(self, new_parent):
        """Duplicate the attribute with a new parent.

        :param new_parent: the new parent to attached the duplicated attribute
            too
        :type new_parent: :class:`~srpt.Schema.SchemaObject` or :class:`~srpt.Schema.SchemaRelation`

        :returns: the duplicated attribute
        :rtype: :class:`~.SchemaAttribute`
        """

        attrs = {}
        for name in self.__attr_names:
            attrs[name] = getattr(self, name)
        return SchemaAttribute(self.name, new_parent, attrs)

    def __str__(self):
        return 'SchemaAttribute[name=%s, parent=%r type=%s]' % (self.name, self.parent, self.type)

    def __repr__(self):
        return 'ScAttr[name=%s, type=%s]' % (self.name, self.type)


# A relation represented in the schema
class SchemaRelation(object):
    """
    :var ignore: should this relation be ignored?
    :var name: this relation's name
    :var id: unique id, a tuple equal to *(self.name, self.source_type, self.target_type)*
    :var source_type: the type of the source object
    :var target_type: the type of the target_object
    :var reflection: the name of the relation that is the reflection of this one,
        if one exists
    :var attributes: (Dict[attribute.name => :class:`~.SchemaAttribute`])
        all the attributes of this object
    """

    def __init__(self, schema, name, source_type, target_type, attrs):
        """Create SchemaRelation object. It automatically updates the *out_relations*
        of the source object and the *in_relations* of the target object. Also,
        updates the schema's *relations* and *object_and_relations*.

        :param schema: the schema this relation belongs to
        :param name: the name of the relation
        :param source_type: the type of the source object (:class:`SchemaObject.type <.SchemaObject>`)
            which is an index into :class:`Schema.objects <.Schema>`
        :param target_type: the type of the target object (:class:`SchemaObject.type <.SchemaObject>`)
            which is an index into :class:`Schema.objects <.Schema>` 
        :param attrs: the attributes to use in setting up the object

                * ignore - should this object be ignored
                * reflection - the reflection for this relation that goes
                  between target and source (if one exists)
        """

        self.ignore = attrs.get('ignore', False)
        self.schema = schema

        self.name = name
        self.source_type = source_type
        self.target_type = target_type
        self.type = (self.name, self.source_type, self.target_type)
        self.reflection = attrs.get('reflection', None)
        self.attributes = {}

        self.id = (self.name, self.source_type, self.target_type)

        if self.target_type in self.schema.objects and self.source_type in self.schema.objects:
            self.schema.objects[self.target_type].in_relations[self.id] = self
            self.schema.objects[self.source_type].out_relations[self.id] = self

        self.schema.relations[self.id] = self
        self.schema.relation_names.append(self.name)
        self.schema.objects_and_relations.append(self)

    def __str__(self):
        s = 'SchemaRelation[name=%s, source_type=%s, target_type=%s, reflection=%s' % (
        self.name, self.source_type, self.target_type, self.reflection)
        s = '%s\n\t attribs=[%s]' % (s, '\n\t\t  '.join(map(repr, self.attributes.values())))
        s = '%s]' % (s)
        return s

    def __repr__(self):
        return 'ScRel[%s: %s -> %s]' % (self.name, self.source_type, self.target_type)