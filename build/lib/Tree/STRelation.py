
from utilities.utils import instanceLoader


class STRelation(object):
    """Represents a spatio-temporal relationship between two
     :class:`STObjects <srpt.STObject.STObject>`.

     :var name: the relationships name
     :var start_time: the time the relationship starts to exist
     :var end_time: the time the relationship ceases to exist
     :var source: the source :class:`~srpt.STObject.STObject`
     :var target: the target :class:`~srpt.STObject.STObject`
     :var reflection: the reflection :class:`~srpt.STRelation.STRelation` of this relationship
     :var attributes: (Dict[attribute.id => :class:`~srpt.STAttribute.STAttribute`]) the attributes of this object
     """

    __slots__ = (
    'graph', 'id', 'name', 'source', 'target', 'attributes', 'source', 'target', 'reflection', 'type', 'start_time',
    'end_time', 'source_id', 'target_id')

    def __init__(self, rel_id, rel_type, graph, source, target,
                 start_time=None, end_time=None, reflection=None):
        """Create a directed relationship between two objects.

        :param rel_id: unique id
        :param rel_type: the name/type of this relationship
        :param graph: the graph this relationship belongs to
        :param source: the source :class:`~srpt.STObject.STObject`
        :param target: the target :class:`~srpt.STObject.STObject`
        :param start_time: the time the relationship starts to exist. if not 
            given the relationship is assumed to start at the earliest time
            both the source and target objects exist
        :param end_time: the time the relationship stops existing. if not given
            the relationship is assumed to last until either the source or target
            ceases to exist
        :param reflection: if this relationship is a reflection (a relationship
            going from target to source, the opposite direction of this one), 
            then this is the :class:`~srpt.STRelation.STRelation` that is that
            reflection
        """

        self.graph = graph
        self.id = rel_id
        self.name = rel_type
        self.source = source
        self.target = target

        # track our attributes
        self.attributes = {}

        # track the relations
        self.source.out_relations.add(self)
        self.target.in_relations.add(self)
        self.reflection = reflection

        # a relations type is uniquely defined by this triple
        self.type = (rel_type, self.source.type, self.target.type)

        # the graph is tracking objects in multiple ways, update them all
        self.graph.relations[self.id] = self
        self.graph.relations_by_type[self.type].append(self)

        # if start time is not given then the relationship starts
        # at the earliest time both the source and target object exist
        self.start_time = start_time if start_time is not None else max(source.start_time, target.start_time)

        # if end time is not given then the relationship ends
        # as soon as one object ceases to exist
        self.end_time = end_time if end_time is not None else min(source.end_time, target.end_time)

    def _setGraph(self, graph):
        """During unpickling this is called to attach the relationship to the
        graph it belongs to.

        :param graph: the graph to attach this relationship to
        """

        self.graph = graph
        self.source = self.graph.objects[self.source_id]
        self.target = self.graph.objects[self.target_id]
        self.type = (self.name, self.source.type, self.target.type)
        self.source.out_relations.add(self)
        self.target.in_relations.add(self)

    def __reduce__(self):
        args = (self.id, self.name, self.start_time, self.end_time, self.source.id, self.target.id)
        return (instanceLoader, (self.__class__, args))

    def _initWithArgs(self, args):
        """Callback for unpickling with :func:`~utilities.utils.instanceLoader`."""

        self.id = args[0]
        self.name = args[1]
        self.start_time = args[2]
        self.end_time = args[3]
        self.source_id = args[4]
        self.target_id = args[5]

        self.attributes = {}

    def __str__(self, indent=0):
        """
        if hasattr(self, 'reflection'):
            rel = ', reflection=%r' % self.reflection.id
        else:
        """
        rel = ''
        s = 'STRelation[id=%s, name=%s, source=%r, target=%r%s' % (self.id, self.name, self.source, self.target, rel)
        s = '%s\n\t    attribs=[%s]' % (s, '\n\t\t      '.join(map(repr, self.attributes.values())))
        s = '%s]' % (s)
        return s

    def __repr__(self):
        return 'STRel[%s(%s): %r -> %r]' % (self.name, self.id, self.source, self.target)