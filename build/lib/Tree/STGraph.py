from collections import defaultdict

from utilities.utils import instanceLoader


class STGraph(object):
    """
    :var id: the graph's id
    :var schema: the schema this graph and all its objects and relationships and
        their attributes conform too
    :var objects: (Dict[object.id => :class:`srpt.STObject.STObject`]) 
        objects in this graph
    :var objects_by_type: (Dict[object.type => List[:class:`srpt.STObject.STObject`] 
        objects in this graph grouped by type
    :var relations: (Dict[relation.id => :class:`srpt.STRelation.STRelation`]) 
        relations in this graph 
    :var relations_by_type: (Dict[relation.type => List[:class:`srpt.STRelation.STRelation`] 
        relations in this graph grouped by type
    :var attributes: (Dict[attribute.id => :class:`srpt.STAttribute.STAttribute`]) 
        attributes in this graph 
    :var attributes_by_type: (Dict[attribute.type => List[:class:`srpt.STAttribute.STAttribute`] 
        attributes in this graph grouped by type
    :var class_label: the class label of this graph, used for training
    :var start_time: the time the graph starts (generally 1)
    :var end_time: the time the graph ends
    """
    __slots__ = ('id', 'type', 'schema', 'objects', 'objects_by_type', 'relations', 'relations_by_type',
                 'attributes', 'attributes_by_type', 'class_label', 'start_time', 'end_time','flood_time')

    def __init__(self, graph_id):
        """
        :param graph_id: unique graph id
        """

        self.id = graph_id
        self.type = 'graph'

        # track all the objects, attributes, and relations by id and type
        self.objects = {}
        self.objects_by_type = defaultdict(list)

        self.relations = {}
        self.relations_by_type = defaultdict(list)

        self.attributes = {}
        self.attributes_by_type = defaultdict(list)

        self.class_label = None
        self.flood_time = None



    def __reduce__(self):
        # note that the item_by_type dicts are pickled as the information is
        # redundant, instead the are recreated during unpickling
        args = (self.id, self.class_label,self.flood_time, self.objects, self.relations, self.attributes)
        return (instanceLoader, (self.__class__, args))

    def _initWithArgs(self, args):
        """Callback for unpickling with :func:`~utilities.utils.instanceLoader`."""
        print("Callling init_with args where we are assigning object type:")
        self.type = 'graph'

        self.id = args[0]
        self.class_label = args[1]
        self.flood_time = args[2]
        self.objects = args[3]
        self.relations = args[4]
        self.attributes = args[5]

        # these dicts aren't pickled since the information is redundant, so
        # we recreate them during unpickling
        self.objects_by_type = defaultdict(list)
        self.relations_by_type = defaultdict(list)
        self.attributes_by_type = defaultdict(list)

        for obj in self.objects.values():
            obj.graph = self
            self.objects_by_type[obj.type].append(obj)

        for rel in self.relations.values():
            # once the relations are loaded we need to tell them which graph
            # they belong to so that they can connect themselves back up
            # to their source and target objects
            rel._setGraph(self)
            self.relations_by_type[rel.type].append(rel)

        for attrib in self.attributes.values():
            # once the attributes are loaded we need to tell them which graph
            # they belong to so that they can attach themselves back to their
            # parent objects/relations
            attrib._setGraph(self)
            self.attributes[attrib.id] = attrib
            self.attributes_by_type[attrib.type].append(attrib)

    def __str__(self):
        s = 'STGraph[id=%s\n class_label=%s\n flood_time=%s' % (self.id,self.class_label,self.flood_time)
        # s += '# schema objs + rel: %s' % (len(self.schema.objects_and_relations))
        s += '\t# graph objs: %s\n' % (len(self.objects))
        s += '\t# graph rels: %s\n' % (len(self.relations))
        s += '\tObjects:\n'
        for obj in self.objects.values():
            s += '\t\t%s' % str(obj).replace('\t\t', '\0x1').replace('\t', '\t\t').replace('\0x1', '\t\t\t')
            s += '\n'

        s += '\n'
        s += '\tRelations:\n'
        for obj in self.relations.values():
            s += '\t\t%s' % str(obj).replace('\t\t', '\0x1').replace('\t', '\t\t').replace('\0x1', '\t\t\t')
            s += '\n'

        s += '\n'
        s += '\tAttributes:\n'
        for obj in self.attributes.values():
            s += '\t\t%s' % repr(obj)
            s += '\n'
        return s

    def __repr__(self):
        return 'STGraph[id=%s, class_label=%s,flood_time=%s, #objs=%s, #rels=%s, #attrs=%s]' % (
        self.id, self.class_label,self.flood_time, len(self.objects), len(self.relations), len(self.attributes))