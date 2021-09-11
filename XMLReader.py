'''
This module provides all the classes and methods responsible for reading
schema from XML and STGraphs from XML.

Loading a Schema object from XML:
schema = SchemaReader.loadFromXML('experiments/shapes/schema.xml')

Loading a STGraph object from XML:
graphs = GraphReader.loadFromXML(schema, 'experiments/shapes/data/numobjects=10_numgraphs=100.xml')
'''

import xml.sax
import sys, os

import traceback
import numpy as np
import re

sys.path.append(os.path.abspath('./'))
from Tree.Schema import Schema, SchemaGraph, SchemaObject, SchemaRelation, SchemaAttribute
from Tree.STGraph import STGraph
from Tree.STObject import STObject
from Tree.STRelation import STRelation
from Tree.STAttribute import STAttribute, DataTypes


def main():
    testLoad()


def testLoad():
    #
    # Demo loading the schema and graphs from XML
    #

    print('------------------ Parse Schema ------------------')

    #schema = SchemaReader.loadFromXML('experiments/shapes/schema.xml')
    # schema = SchemaReader.loadFromXML('tests/xml/flatxml/car_schema.xml')
    schema=SchemaReader.loadFromXML(r"C:\Users\user\Desktop\GraduationProject\Prediction_model\experiments\flood\flood_schema.xml")
    # sys.exit()


    print( '------------------ Schema ------------------')
    print(schema.graph)

    print('Objects:')
    keys = schema.objects.keys()
    keys.sort()
    for obj in (schema.objects[key] for key in keys):
        print(obj)


    print('Relations:')
    types = [rel.id for rel in schema.relations.values()]
    types.sort()

    for rel in (schema.relations[rel] for rel in types):
        print(rel)


    print('Attributes:')
    for key, obj in schema.attributes.items():
        print(key, '->', obj)

    # sys.exit()


    print('------------------ Parse Graph ------------------')
    # graphs = GraphReader.loadFromXML(schema, 'tests/xml/flatxml/cars.xml')
    #graphs = GraphReader.loadFromXML(schema, 'experiments/shapes/test_data_increase/run0/test_file.xml')
    graphs=GraphReader.loadFromXML(schema,r"C:\Users\user\Desktop\GraduationProject\Prediction_model\experiments\flood\flood.xml")
    # sys.exit()
    for graph in graphs.values():
        print(graph)

    sys.exit()


# Thrown when a relation references an illegal object id
class BadRelationException(Exception): pass


class stack(list):
    """A quick and dirty stack like object based on list, simply uses stack
    nomenclature and provides a 'top' attribute (aka peek())"""

    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)

    top = property(lambda self: None if len(self) == 0 else self.__getitem__(-1))
    push = lambda self, v: self.append(v)
    pop = lambda self: None if len(self) == 0 else list.pop(self)


class SimpleElement(object):
    """A plain XML element used mainly in schema loading for elements that don't need
    any special functionality."""

    def __init__(self):
        self.cur_content = ''


class ProtoRelation(SimpleElement):
    """Used when loading relations from schema to hold cross-product construction
    of a relationship."""

    def __init__(self):
        SimpleElement.__init__(self)
        self.attributes = {}


def dereferenceID(objs, obj_id):
    """Handles changing XML specified id's into usable ids.

    :param objs: A dict of [id=>obj] to get an object matching the id from
    :param obj_id: The object id from XML which needs to be converted

    :returns: The object from objs whose id is obj_id
    """

    try:
        obj_id = int(obj_id)
    except ValueError:
        pass

    return objs[obj_id]


class ErrorHandler(xml.sax.ErrorHandler):
    def error(self, e):
        print('Error')
        raise e

    def fatalError(self, e):
        print('fatalError')
        raise e

    def warning(self, e):
        print('warning')
        raise e


class GraphReader(xml.sax.ContentHandler):
    """This class is responsible for reading a data file and using a schema to
    instantiate all the objects relationships, and attributes in the data file
    building up a graph object
    """

    @staticmethod
    def loadFromXML(schema, filename, as_iter=False, keep_field_data=False):
        """ Load an STGraph object from "filename". The graph must conform to
        the given schema object.

        :param schema: The schema for the graph
        :param filename: The filename of the XML file to load the graph from
        :param as_iter: If True, instead of returning a dict[graph_id=>graph],
            return an iterator over the graphs
        :param keep_field_data: If True, after processing the field data don't
            discard it but leave it in the STAttribute object

        :returns: A dictionary [graph.id => graph] or an iterator
            if as_iter==True
        """
        parser = xml.sax.make_parser()
        builder = GraphReader(schema, as_iter=as_iter, keep_field_data=keep_field_data)
        builder.filename = filename
        parser.setContentHandler(builder)
        parser.setErrorHandler(ErrorHandler())

        try:
            parser.parse(open(filename))
        except Exception:
            traceback.print_exc()
            if hasattr(builder, 'locator'):
                print('XML: %s\n\tLine: %s' % (builder.locator.filename, builder.locator.getLineNumber()))
            if hasattr(builder, 'cur_element') and builder.cur_element is not None:
                name, raw_attrs, attrs = builder.raw_cur_elem
                print('Raw Element:\n\t%s => %s' % (name, attrs))
                print('Element Stack:')
                for idx in range(len(builder.cur_element) - 1, -1, -1):
                    e = builder.cur_element[idx]
                    print('\t%i: %r(%r)' % (idx, e, e.st if hasattr(e, 'st') else None))
            sys.exit(-1)
        return builder.graphs

    #
    def __init__(self, schema, as_iter, keep_field_data):
        """Creates a GraphReader object to be use by a SAX Parser. Use
        :meth:`~.GraphReader.loadFromXML` instead of creating a
        GraphReader object directly.

        :param schema: The schema to use while loading the STGraph
        :param as_iter: If True, don't load the graphs all at once but make
            graph_reader.graphs an iterator instead of a dict
        :param keep_field_data: If True, keep the field data after its been
            processed instead of discarding it

        """
        xml.sax.ContentHandler.__init__(self)

        self.schema = schema
        self.as_iter = as_iter
        self.keep_field_data = keep_field_data
        self.cur_object = None
        self.cur_relation = None
        self.cur_attribute = None

        self.cur_element = stack()
        self.graphs = {}

        self._done = False

    def setDocumentLocator(self, locator):
        """Setting the document locator provides line numbers from the XML files
        when an error occurs

        :param locator: the document locator to use during parsing
        """

        xml.sax.ContentHandler.setDocumentLocator(self, locator)
        self.locator = locator
        self.locator.filename = self.filename

    def getCurLocation(self):
        """Returns the current location as "filename:linenumber" if a document
        locator has been set.

        :returns: Current location in XML file.
        """

        if hasattr(self, 'locator'):
            s = '%s:%s' % (self.locator.filename, self.locator.getLineNumber())
        else:
            s = ''
        return s

    def skippedEntity(self, name):
        # print '=========> Skipping[%s]' % name
        pass

    def endElement(self, name):
        """Callback from SAX parser when the closing tag of an element is found.

        :param name: the name of the closing tag/element
        """

        # all xml tags are dropped to lower case to help prevent errors
        name = str(name).lower()

        # print '=========> endElem:%s' % name

        if self._done: return

        # pop off the current XML element/tag we are working with
        cur_elem = self.cur_element.pop()
        if cur_elem:
            cur_elem.cur_content = str(cur_elem.cur_content)

        if isinstance(cur_elem, ObjectBuilder):
            pass

        elif isinstance(cur_elem, RelationBuilder):
            pass

        elif isinstance(cur_elem, AttributeBuilder):
            cur_elem._setup(cur_elem.cur_content)
            cur_elem._finalize()

        elif isinstance(cur_elem, FieldData):
            cur_elem._setup(cur_elem.cur_content)
            cur_elem._finalize()

        elif name == 'timestep':
            attrib = cur_elem.attrib
            try:
                attrib.addFieldTimestep(cur_elem)
            except AttributeError:
                # most likely adding a timestep to an ignored object
                pass

        elif isinstance(cur_elem, GraphBuilder):
            cur_elem._setup(cur_elem.cur_content)
            self.graph._fixIds()
            self.graph._fixRelations()
            self.graph._finishLoadingGraph()

            # if self.as_iter:
            #    yield self.graph

            self.graph = None

        if cur_elem:
            try:
                del cur_elem.cur_content
            except AttributeError:
                pass
            del cur_elem

    def startElement(self, name, raw_attrs):
        """Callback for the SAX parser when the start of a element is found.

        :param name: element/tag name
        :raw_attrs: the element/tag attributes from the xml
        """
        if self._done: return

        # all xml tags are dropped to lower case to help prevent errors
        name = str(name).lower()

        # print '=========> startElement:%s' % name

        # Each different kind of item: object, relation, attribute, graph, etc.
        # has its own builder object which are used to actually construct the
        # ST item.

        # the attrs object is not quite a full dict, so copy the info out into
        # an actual dict and perform some value sanitization
        attrs = {}
        for attr in raw_attrs.keys():
            attr_name = attr.replace('-', '_').lower()
            attr_value = str(raw_attrs.get(attr)).lower()
            if attr_value in ('true', 'false'):
                attr_value = bool(attr_value)
            else:
                try:
                    raw_value = attr_value
                    attr_value = int(attr_value)
                    if '.' in raw_value:
                        attr_value = float(attr_value)
                except ValueError:
                    # failed to convert to number
                    pass
            attrs[attr_name] = attr_value
        self.raw_cur_elem = (name, raw_attrs, attrs)

        # tag starts on object
        if name in self.schema.objects:
            self.cur_object = ObjectBuilder(self.graph, self.graph.st, self.schema, name, attrs)
            self.cur_element.push(self.cur_object)
            self.cur_object.cur_content = ''

        # tag starts a relationship
        elif name in self.schema.relation_names:
            try:
                self.cur_relation = RelationBuilder(self.graph, self.graph.st, self.schema, name, attrs)
            except BadRelationException:
                self.cur_relation = SimpleElement()
                self.cur_relation.st = SimpleElement()
                self.cur_relation.st.attributes = []
                self.cur_relation.ignore = True
            self.cur_element.push(self.cur_relation)
            self.cur_relation.cur_content = ''

        # tag is a timestep in an attribute
        elif name == 'timestep':
            timestep = SimpleElement()
            timestep.attrib = self.cur_element.top
            # size and corner don't exist for each timestep of field data objects, as they
            # are the same for all timesteps
            if isinstance(timestep.attrib, AttributeBuilder):
                timestep.size = map(int, attrs['size'].split(','))
                timestep.corner = map(int, attrs['corner'].split(','))
            timestep.datasource = attrs.get('datasource', None)
            timestep._line = self.getCurLocation()
            self.cur_element.push(timestep)

        # tag is a field data object
        elif name in self.schema.fielddata:
            fielddata = FieldData(self.graph, self.graph.st, self.schema, name, attrs)
            self.cur_attribute = fielddata
            self.cur_element.push(fielddata)
            self.cur_attribute.cur_content = ''

        # tag is an attribute inside an object or relation
        elif len(self.cur_element) and name in self.cur_element.top.st.attributes:
            parent = self.cur_element.top
            attrs['keep_field_data'] = self.keep_field_data
            self.cur_attribute = AttributeBuilder(self.graph, self.graph.st, self.schema, name, parent.st, attrs)
            self.cur_element.push(self.cur_attribute)
            self.cur_attribute.cur_content = ''

        # tag is the beginning of a graph
        elif name == 'graph':
            self.graph = GraphBuilder(self.schema, attrs)
            if self.graph.st.id not in self.graphs:
                self.graphs[self.graph.st.id] = self.graph.st
            else:
                line_number = '???'
                if hasattr(self, 'locator'):
                    line_number = self.locator.getLineNumber()
                sys.stderr.write('Duplicate graph id [%s] found at line %s\n' % (self.graph.st.id, line_number))
                #sys.exit(-1)
                pass

            self.cur_element.push(self.graph)

        elif name == 'graphs':
            pass

        else:
            # If no matching schema element was found then throw an error, we don't
            # want to silently handle parsing errors. If something is wrong in the XML
            # it needs to be fixed.
            if hasattr(self, 'locator'):
                print('Invalid element name "%s" on line %s not found in schema.\n' % (
                name, self.locator.getLineNumber()))
                sys.stderr.write('Invalid element name "%s" on line %s not found in schema.\n' % (
                name, self.locator.getLineNumber()))
            sys.exit(-1)

    def characters(self, content):
        """Callback from SAX parser when characters are read between
        the start and end tag of an element.

        :param content: the characters read
        """

        if self._done: return

        # if it wasn't just whitespace add append the content
        if content.strip():
            self.cur_element.top.cur_content += content
            # print '===> Content[%s] => [%s]' % (content[:50], self.cur_element.top.cur_content[:50])


class GraphBuilder(object):
    """Builds a STGraph object from the info loaded from the XML."""

    def __init__(self, schema, attrs=None):
        """Create a GraphBuilder that will make a STGraph tha confirms to the
        given schema.

        :param schema: the schema to conform to
        :param attrs: dictionary of XML tag/element attributes
        """

        self.st = STGraph(attrs.pop('id'))
        # print 'Graph:', self.st.id
        self.st.schema = schema

        # add attribute templates from the schema
        # print self.st.schema.attributes.keys()
        for attr in self.st.schema.graph.attributes.values():
            self.st.attributes[attr.name] = None

        # turn XML attribs into our own attribs
        if attrs is not None:
            for attr, value in attrs.items():
                if attr == 'class_label':
                    temp=list(self.st.schema.class_labels)
                    if temp:
                        value = type(temp[0])(value)
                setattr(self.st, attr, value)

        # track XML text content
        self.cur_content = ''

        # used for tracking autogenerated id's and ensuring they don't overlap
        # XML supplied id's
        self.next_id = 1
        self.max_id = 1

        self.field_data = {}

    def _setup(self, content):
        """Prepares object."""
        # a graph doesn't have any setup to perform
        pass

    def _nextid(self, id):
        """Trys to convert the given id into a usable id, if it fails the next
        unique id that can be assigned to an object or relationship is returned.

        :returns: a unique id
        """

        try:
            id = int(id)
        except (ValueError, TypeError):
            pass

        if id is None:
            id = (self.next_id,)
            self.next_id += 1
        if type(id) is str and id[0] == '#':
            id = (id, self.next_id)
            self.next_id += 1
        elif type(id) is int:
            self.max_id = max(id, self.max_id)

        return id

    def _fixIds(self):
        """Some id's were auto-generated and others were given in the file,
        adjust the auto-generated id's so that they don't over lap the given
        id's, this only applies for int id's. Str id's don't need adjusting as
        no str id's are auto-generated.
        """

        new_max_id = 0
        for group in (self.st.objects, self.st.relations, self.st.attributes):
            for key, obj in group.items():
                # print key, obj
                if obj is None:
                    del group[key]
                    continue

                if type(obj.id) is tuple and len(obj.id) == 1:
                    # print 'Fixing: %s=>%s : %s' % (key, obj.id, obj)
                    # not only do we adjust the object's id, we must also update the
                    # dict holding the object
                    new_id = obj.id[0] + self.st.schema.max_id

                    # we only touch the dicts if the key was the id
                    if key == obj.id:
                        del group[key]
                        group[new_id] = obj
                    obj.id = new_id
                    new_max_id = max(new_max_id, new_id)
        self.st.schema.max_id = new_max_id + 1
        self.next_id = new_max_id

    def _finishLoadingGraph(self):
        """Callback called at the end of a graph loading."""

        # print '\t', len(self.st.objects), len(self.st.relations)
        pass

    def _fixRelations(self):
        """Create reflections for relations that should have them but don't."""

        # print
        # print 'Fixing Relations:'

        for rel in self.st.relations.values():
            # print
            # print self.st.relations_by_type.keys()
            # determine the type of the reflection of this relation (if it doesn't have one it becomes None)
            rel.reflection = self.st.schema.relations[rel.type].reflection

            # if there should be a reflection create it
            if rel.reflection:
                # check for the existence of the reflection relation
                rel_type = (rel.reflection, rel.target.type, rel.source.type)
                # rel_id = (rel.reflection, rel.target.id, rel.source.id)
                # print 'Reflect:', rel_type

                # get the relations of the correct type
                rels = self.st.relations_by_type.get(rel_type, [])
                # print 'Rels:', rels
                # find the relation with the target and source switched
                rels = [arel for arel in rels
                        if arel.target.id == rel.source.id and
                        arel.source.id == rel.target.id]
                # print 'Matching Rels:', rels
                # print 'Looking for:', rel_id

                # if we found a matching relation then just set the reflection
                # to that relation
                if len(rels) > 0:
                    rel.reflection = rels[0]
                    rels[0].reflection = rel

                # relation doesn't exist, we need to create it
                else:
                    # print 'New Reflection: %r, %r, %r' % (rel.reflection, rel.target.id, rel.source.id)
                    attrs = dict(source=rel.target.id, target=rel.source.id)
                    new_rel = RelationBuilder(self, rel.graph, self.st.schema, rel.reflection, attrs)
                    new_rel.st.id = self._nextid(None)[0]
                    # print '\tid=%s' % (new_rel.id)
                    new_rel.st.attributes = rel.attributes
                    rel.reflection = new_rel.st
                    new_rel.st.reflection = rel
                    # print new_rel.st


# Base classes for all objects, relations and attributes
class ObjectBuilder(object):
    """Builds a :class:`~srpt.STObject.STObject` from XML."""

    def __init__(self, builder, graph, schema, obj_type, attrs):
        """Creates a STObject of type obj_type from XML that belongs to the
        given graph and confirms to the given schema.

        :param builder: the graph builder who built this object's parent STGraph
        :param graph: this object's parent STGraph
        :param schema: the schema to conform to
        :param obj_type: the type of this object
        :param attrs: any xml attrs to add to this object
        """

        self.builder = builder
        obj_id = builder._nextid(attrs.pop('id', None))
        self.st = STObject(obj_id, obj_type, graph)
        # print '\t', obj_id, obj_type

        self.schema = schema

        # add attribute templates from the schema
        obj = self.schema.objects.get(obj_type)
        for attr in obj.attributes.values():
            self.st.attributes[attr.name] = None

        # turn XML attribs into our own attribs
        for attr, value in attrs.items():
            setattr(self.st, attr, value)


class RelationBuilder(object):
    """Builds a :class:`~srpt.STRelation.STRelation` from XML."""

    def __init__(self, builder, graph, schema, rel_type, attrs):
        """Creates a STRelation of type rel_type from XML that belongs to the
        given graph and confirms to the given schema.

        :param builder: the graph builder who built this relation's parent STGraph
        :param graph: this relations's parent STGraph
        :param schema: the schema to conform to
        :param rel_type: the type of this relation
        :param attrs: any xml attrs to add to this relation
        """

        self.builder = builder
        # if no id is set auto-assign one
        rel_id = builder._nextid(attrs.pop('id', None))

        # get source and target objects
        try:
            source = dereferenceID(graph.objects, attrs.pop('source'))
            #print("source of the relation {0}".format(source))
            target = dereferenceID(graph.objects, attrs.pop('target'))
            #print("target of the relation {0}".format(target))
        except KeyError:
            raise BadRelationException()

        self.st = STRelation(rel_id, rel_type, graph, source, target)
        # print '\t', rel_id, rel_type, source.type, target.type
        self.schema = schema
        self.st.reflection = attrs.pop('reflection', None)

        # make any XML attrs our own
        for attr, value in attrs.items():
            setattr(self.st, attr, value)

        # populate with attribute templates
        rel = self.schema.relations[self.st.type]
        for attr in rel.attributes.values():
            self.st.attributes[attr.name] = None


class FieldData(object):
    """Builds a FieldData object from XML."""

    def __init__(self, builder, graph, schema, name, attrs):
        """Creates a field data object named `name` from XML that belongs to the
        given graph and confirms to the given schema.

        :param builder: the graph builder who built this field data's parent STGraph
        :param graph: this field data's parent STGraph
        :param schema: the schema to conform to
        :param name: the type of this relation
        :param attrs: any xml attrs to add to this field data
        """

        self.builder = builder

        self.schema = schema
        self.graph = graph
        self.builder.field_data[name] = self
        self.timesteps = []

        self.size = map(int, attrs['size'].split(','))
        self.type = schema.fielddata[name]

    def _setup(self, value):
        # store the value for later conversion
        self.value = value

    def addFieldTimestep(self, elem):
        elem.content = elem.cur_content
        self.timesteps.append(elem)

    def _finalize(self):
        # convert timesteps into values
        for i, step in enumerate(self.timesteps):
            vals = map(float, step.content.split(','))

            size = list(self.size)
            if 'Vector' in self.type:
                size.append(len(size))
            vals = np.array(vals, dtype=float).reshape(size)
            self.timesteps[i] = vals


class AttributeBuilder(object):
    """Builds a :class:`~srpt.STAttribute.STAttribute` from XML."""

    bracket_pattern = re.compile('\]\s*,\s*\[')

    def __init__(self, builder, graph, schema, name, parent, attrs):
        """Creates a STAttribute of named `name` from XML that belongs to the
        given graph and confirms to the given schema.

        :param builder: the graph builder who built this attribute's parent
            object/relationship
        :param graph: this attributes parent STGraph
        :param schema: the schema to conform to
        :param name: the name of this attribute
        :param parent: the parent STObject or STRelation this attribute belongs to
        :param attrs: any xml attrs to add to this relation
        """
        self.builder = builder

        # grab an ID if we don't already have one
        attr_id = builder._nextid(attrs.pop('id', None))

        if isinstance(parent, STGraph):
            schema_attribute = schema.graph.attributes[name]
        else:
            schema_attribute = schema.attributes[parent.type][name]

        attr_type = getattr(schema_attribute, 'type', 'Discrete')
        #print("attr_type")
        #print(attr_type)
        if 'Float' in attr_type:
            self.dtype = np.float
        elif 'Int' in attr_type:
            self.dtype = np.int
        else:
            self.dtype = np.str

        self.st = STAttribute(attr_id, name, parent, graph, data_type=attr_type)
        print ('\t\t', self.st.id, self.st.name, self.st.data_type)
        self.schema = schema
        self.schema_attribute = schema_attribute
        # make any XML attrs our own
        for attr, value in attrs.items():
            setattr(self.st, attr, value)

        self.timesteps = []

    def _setup(self, value):
        if self.st is None:
            return

            # store the value for later conversion
        self._value = value

    def addFieldTimestep(self, elem):
        elem.content = elem.cur_content
        self.timesteps.append(elem)

    def _finalize(self):
        # print 'Finalizing:', self.st
        if self.st is None:
            return

            # set up the attribute's value
        # if isinstance(self.st.parent, STGraph):
        #    parent_type= 'graph'
        # else:
        #    parent_type = self.st.parent.type

        attr_type = getattr(self.schema_attribute, 'type', DataTypes.Discrete)  # @UndefinedVariable

        if len(self.timesteps) == 0:
            val = convert_value(self._value, attr_type)
            #print('Converting: [%s][%s][%s]=[%s=%s]' % (self.st.name, attr_type, self.st.value, type(val), val))
            self.st.value = val
        else:
            # convert timesteps into values
            st_value = []
            st_mask = []
            st_corners = []
            st_linenumbers = []
            for idx, step in enumerate(self.timesteps):
                if self.st.datalocation == 'self':
                    try:
                        if ',' in step.content:
                            splitter = ','
                        else:
                            splitter = ' '

                        vals, mask = step.content.split(';')
                        vals = vals.strip()
                        vals = map(float, vals.split(splitter))

                        mask = mask.strip()
                        mask = map(float, mask.split(splitter))

                        size = list(step.size)
                        if 'Vector' in attr_type:
                            # print 'Vector:', attr_type, size, len(size)
                            # if len(size) == 3:
                            #    if size[2] == 1:
                            #        size.append(2)
                            #    else:
                            #        size.append(3)
                            # else:
                            #    size.append(len(size))
                            size.append(len(size))
                        vals = np.array(vals, dtype=self.dtype).reshape(size)
                        mask = np.array(mask, dtype=bool).reshape(step.size)

                        # print '\t', vals.shape
                        # print '\t', mask.shape
                    except Exception:
                        vals, mask = step.content.split(';')
                        print('Attribute[name=%s, type=%s]' % (self.st.name, self.st.data_type))
                        print('attrib-shape:', step.size)
                        print('\ttotal:', np.prod(step.size))
                        size = list(step.size)
                        if 'Vector' in attr_type:
                            size.append(len(size))
                        print('size:', size)
                        print('vals:', type(vals))
                        print('mask:', type(mask))

                        try:
                            vals = vals.strip()
                            vals = vals.split(splitter)
                            print('Len(vals):', len(vals))
                        except:
                            pass

                        try:
                            mask = mask.strip()
                            mask = mask.split(splitter)
                            print('Len(mask):', len(mask))
                        except:
                            pass

                        raise
                elif self.st.datalocation == 'internal':
                    mask = map(float, step.content.split(','))

                    size = list(step.size)
                    if 'Vector' in attr_type:
                        size.append(len(size))
                    mask = np.array(mask, dtype=bool).reshape(step.size)

                    # we have to slice into the field data array, so build up the slice
                    # objects list
                    slices = []
                    for i, j in zip(step.corner, step.size):
                        slices.append(slice(i, i + j, None))
                    if len(size) > step.size:
                        slices.append(slice(None, None, None))

                    fielddata = self.builder.field_data[step.datasource]
                    vals = fielddata.timesteps[idx][slices]
                else:
                    raise ValueError('Unknown or unimplemented data location: %s' % (self.datalocation,))

                st_value.append(vals)
                st_mask.append(mask)
                st_corners.append(step.corner)
                st_linenumbers.append(step._line)
            self.st.mask = st_mask
            self.st.value = st_value
            self.st.corners = st_corners
            self.linenumbers = st_linenumbers
        self.st._calcStats(getattr(self.schema, 'do_fields', True))


def convert_value(value, attr_type=DataTypes.Discrete):  # @UndefinedVariable
    """Converts the value load from XML into the correct python type based upon
    the attribute type.

    :param value: the value to convert
    :param attr_type: the attribute type that specifies how the value should
        be converted
    """

    try:
        timestep_splitter = ','
        if 'Float' in attr_type or 'LatLong' in attr_type:
            conversion_function = float

        elif 'Int' in attr_type:
            conversion_function = int
        elif 'Coord' in attr_type:
            timestep_splitter = ';'
            conversion_function = lambda v: map(float, v.split(','))
        else:
            conversion_function = str
        print(conversion_function)
            # if its an array we split on commas
        if 'Array' in attr_type:

            value = value.split(timestep_splitter)
            value = map(conversion_function, value)
            value=list(value)
            if 'Discrete' not in attr_type and 'Str' not in attr_type:
                value = np.array(value)
        else:
            value = conversion_function(value)

    except NameError:
        # conversion function not found
        pass
   #print("the value returned from conversion function:")
    #print(value)
    #print("type of the value returned from the conversion function {0}".format(type(value)))
    return value


class SchemaReader(xml.sax.ContentHandler):
    """Parses a schema XML file and builds a :class:`~srpt.Schema.Schema` object."""

    @staticmethod
    def loadFromXML(filename, **kwargs):
        """Loads the schem from the XML file *filename* and returns a schema
        object.

        :param filename: the xml file that contains the schema
        :param \*\*kwargs: keyword arguments that will be set on this object. I.e.
            schema.key=value for each key-value pair

        :returns: A schema objected load from XML
        """

        parser = xml.sax.make_parser()

        schema = Schema()

        reader = SchemaReader(schema)
        reader.filename = filename

        # set the attributes given in kwargs
        for k, v in kwargs.items():
            setattr(reader, k, v)

        # keep_proto_relations must exist, so default it to false
        reader.keep_proto_relations = getattr(reader, 'keep_proto_relations', False)
        if reader.keep_proto_relations:
            schema.proto_relations = []

        # setup the SAX parser and parse the xml file
        parser.setContentHandler(reader)
        try:
            parser.parse(open(filename))
        except Exception:
            traceback.print_exc()
            if hasattr(reader, 'locator'):
                print('XML: %s\n\tLine: %s' % (reader.locator.filename, reader.locator.getLineNumber()))
            sys.exit(-1)
        return schema

    def __init__(self, schema):
        """Creates a SchemaReader object used by the SAX parser to build a
        schema object from xml. Use :meth:`~.SchemaReader.loadFromXML`
        instead of creating the object directly.
        """

        self.schema = schema
        self.cur_object = None
        self.cur_relation = None
        self.cur_attribute = None
        self.cur_element = stack()
        self.split_relations = True

    def setDocumentLocator(self, locator):
        """Setting the document locator provides line numbers from the XML files
        when an error occurs

        :param locator: the document locator to use during parsing
        """

        xml.sax.ContentHandler.setDocumentLocator(self, locator)
        self.locator = locator
        self.locator.filename = self.filename

    def getCurLocation(self):
        """Returns the current location as "filename:linenumber" if a document
        locator has been set.

        :returns: Current location in XML file.
        """

        if hasattr(self, 'locator'):
            s = '%s:%s' % (self.locator.filename, self.locator.getLineNumber())
        else:
            s = ''
        return s

    def startElement(self, name, raw_attrs):
        """Callback for the SAX parser when the start of a element is found.

        :param name: element/tag name
        :raw_attrs: the element/tag attributes from the xml
        """

        name = str(name).lower()
        attrs = {}
        for attr in raw_attrs.keys():
            attr_value = str(raw_attrs.get(attr)).lower()
            if attr_value in ('true', 'false'):
                attr_value = bool(attr_value)
            else:
                try:
                    raw_value = attr_value
                    attr_value = int(attr_value)
                    if '.' in raw_value:
                        attr_value = float(attr_value)
                except ValueError:
                    # failed to convert to number
                    pass
            attrs[str(attr)] = attr_value

        if name == 'object':
            self.cur_object = SchemaObject(self.schema, attrs)
            self.cur_element.push(self.cur_object)

        elif name == 'relation':
            # begin gathering the info needed to create the relation
            rel_name = attrs.pop('type')
            source_type = attrs.pop('source_type', None)
            source_type = source_type if source_type is not None else self.cur_element.top.type
            target_type = attrs.pop('target_type')

            # if the target_type or source_type has a comma in them then it is a cross-product
            # construction, which we can't do until the relationship has been fully loaded with
            # attributes, so create a ProtoRelation and hold off creating things till the end
            # of the relation
            if ',' in source_type or ',' in target_type:
                self.cur_relation = ProtoRelation()
                self.cur_relation.id = rel_name
                self.cur_relation.name = rel_name
                self.cur_relation.source_type = source_type
                self.cur_relation.target_type = target_type
                self.cur_relation.attrs = attrs
            else:
                # a normal relation, so just create it like always
                self.cur_relation = SchemaRelation(self.schema, rel_name, source_type, target_type, attrs)
            self.cur_element.push(self.cur_relation)

        elif name == 'attribute':
            no_schema = False

            attr_name = attrs.pop('name')
            parent = self.cur_relation if self.cur_relation else self.cur_object

            # ProtoRelations shouldn't have their attribubutes added to the schema, we will
            # take care of this manually later
            if isinstance(parent, ProtoRelation):
                no_schema = True

            self.cur_attribute = SchemaAttribute(attr_name, parent, attrs, no_schema=no_schema)
            self.cur_element.push(self.cur_attribute)

        elif name == 'field-data':

            data_name = attrs['name']
            parts = attrs['type'].split('-')
            parts = map(str.capitalize, parts)
            data_type = DataTypes.get(''.join(parts))
            self.schema.fielddata[data_name] = data_type

        elif name == 'graph':
            g = SchemaGraph(self.schema, attrs)
            self.cur_element.push(g)

        elif name == 'attrib':
            cur_elem = SimpleElement()
            cur_elem.__dict__.update(attrs)
            self.cur_element.push(cur_elem)
            print("this is inside if name=='attrib'")
            print("the current element is:")
            print(attrs)
        elif name == 'schema':
            self.cur_element.push(self.schema)

        if not hasattr(self.cur_element.top, 'cur_content'):
            self.cur_element.top.cur_content = ''

    def endElement(self, name):
        """Callback from SAX parser when the closing tag of an element is found.

        :param name: the name of the closing tag/element
        """

        name = str(name).lower()

        cur_elem = self.cur_element.pop()

        if name == 'object':
            self.cur_object = None
        elif name == 'attribute':
            self.cur_attribute = None
        elif name == 'relation':
            # ProtoRelations are cross-product constructions, so we need to
            # expand things
            if self.keep_proto_relations:
                self.schema.proto_relations.append(self.cur_relation)

            if isinstance(self.cur_relation, ProtoRelation):
                source_types = self.cur_relation.source_type.split(',')
                target_types = self.cur_relation.target_type.split(',')

                # perform the cross-product construction
                for source_type in source_types:
                    for target_type in target_types:
                        relation = SchemaRelation(self.schema, self.cur_relation.name, source_type, target_type,
                                                  self.cur_relation.attrs)
                        for attribute in self.cur_relation.attributes.values():
                            attribute.dup(relation)

            self.cur_relation = None

        elif name == 'schema':
            self.makeReflections()
        elif name == 'graph':
            pass
        elif name == 'attrib':
            val = cur_elem.cur_content

            attr_type = getattr(cur_elem, 'type', DataTypes.Discrete)  # @UndefinedVariable
            if type(attr_type) is str:
                attr_type = DataTypes.get(''.join(map(str.capitalize, attr_type.split('-'))))

            val = convert_value(val, attr_type)
            setattr(self.cur_element.top, cur_elem.name, val)
            self.cur_element.top.attrs[cur_elem.name] = val

        if hasattr(cur_elem, 'cur_content'):
            del cur_elem.cur_content

    def characters(self, content):
        """Callback from SAX parser when characters are read between
        the start and end tag of an element.

        :param content: the characters read
        """

        content = content.strip()
        if content:
            self.cur_element.top.cur_content += content

    def makeReflections(self):
        """Creates the reflections specified in the schema when a relationship
        gives its reflection as a tag-attribute instead of creating a seperate
        relationship just for the reflection.
        """

        for rel in self.schema.relations.values():
            self.schema.objects[rel.target_type].in_relations[rel.id] = rel
            self.schema.objects[rel.source_type].out_relations[rel.id] = rel

            if rel.reflection and not self.schema.relations.has_key((rel.reflection, rel.target_type, rel.source_type)):
                # print 'making reflection:' , (rel.reflection, rel.target_type, rel.source_type)
                reflected_data = {}
                reflected_data['reflection'] = rel.name
                reflected = SchemaRelation(self.schema, rel.reflection, rel.target_type, rel.source_type,
                                           reflected_data)
                reflected.attributes = dict(rel.attributes)
                self.schema.objects[reflected.target_type].in_relations[reflected.type] = rel
                self.schema.attributes[reflected.id] = reflected.attributes


if __name__ == '__main__':
    main()
