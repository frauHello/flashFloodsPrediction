

from utilities.utils import instanceLoader


class STObject(object):
    """A spatio-temporal object stored in STGraph objects.

    :var type: this objects type/name
    :var start_time: the time this object starts to exist
    :var end_time: the time this object ceases to exist
    :var attributes: (Dict[attribute.id => :class:`~srpt.STAttribute.STAttribute`]) the attributes of this object
    :var in_relations: (Set[:class:`~srpt.STRelation.STRelation`]) the set of relations where this object is the target 
    :var out_relations: (Set[:class:`~srpt.STRelation.STRelation`]) the set of relations where this object is the source 

    """

    __slots__ = ('id', 'graph', 'type', 'attributes', 'in_relations', 'out_relations', 'start_time', 'end_time','flood_time')

    def __init__(self, obj_id, obj_type, graph, start_time=None, end_time=None,flood_time=None):
        """Create an STObject object. It will automatically be added to the
        parent graphs objects and objects_by_type attributes.

        :param obj_id: object id, unique in the parent graph
        :param obj_type: objects type
        :param graph: parent graph that this object will be added to
        :param start_time: the time the object first starts to exist. if not given
            assumed to be 1
        :param end_time: the time the object ceases to exist. if not given the
            object is assumed to last until the end of the graph
        """
        self.id = obj_id
        self.graph = graph
        self.type = obj_type
        self.attributes = {}

        # keep track of our incoming and outgoing relations
        self.in_relations = set()
        self.out_relations = set()

        # the graph is tracking objects in multiple ways, update them all
        self.graph.objects[self.id] = self
        self.graph.objects_by_type[self.type].append(self)

        self.start_time = 1 if start_time is None else start_time
        self.end_time = self.graph.end_time if end_time is None else end_time
        self.flood_time=self.graph.flood_time if flood_time is None else flood_time

    def __reduce__(self):
        # prepare the object for pickling
        args = (self.id, self.type, self.start_time, self.end_time)
        return (instanceLoader, (self.__class__, args))

    def _initWithArgs(self, args):
        """Callback for unpickling with :func:`~utilities.utils.instanceLoader`."""

        self.id = args[0]
        self.type = args[1]
        self.start_time = args[2]
        self.end_time = args[3]
        self.flood_time = args[4]
        self.out_relations = set()
        self.in_relations = set()

        self.attributes = {}

    def __str__(self, indent=0):
        tab = '\t' * indent
        s = '%sSTObject[id=%s, type=%s,' % (tab, self.id, self.type)
        attr_tab = '\n%s   ' % ('\t' * (indent + 1))
        item_tab = '\n%s     ' % ('\t' * (indent + 2))
        s = '%s%s attribs=[%s]' % (s, attr_tab, item_tab.join(map(repr, self.attributes.values())))
        s = '%s%s in-rels=[%s]' % (s, attr_tab, item_tab.join(map(repr, self.in_relations)))
        s = '%s%sout-rels=[%s]' % (s, attr_tab, item_tab.join(map(repr, self.out_relations)))
        s = '%s%s]' % (tab, s)
        return s

    def __repr__(self):
        return 'STObj[id=%s, type=%s]' % (self.id, self.type)