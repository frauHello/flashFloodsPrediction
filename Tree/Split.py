from utilities.ContigencyTable import ContingencyTable
import numpy as np
import math



class GraphSplitsDict(dict):
    """A subclass of the normal dict object that is designed to only be used
    with :class:`STGraph.id's <srpt.STGraph.STGraph>` as keys and
    :class:`GraphSplits <.GraphSplit>` as values. However, the normal
    ``iter*`` functions iterator over the STGraphs of the GraphSplits instead of
    the GraphSplits themselves.

    GraphSplitsDicts are used as the *yes* and *no* attributes of a :class:`.Split`.
    """

    def _itervalues(self):
        """Iterates over the GraphSplit objects instead of the STGraphs in
        this object."""

        return dict.values(self)

    def itervalues(self):
        """Iterates over the STGraphs in this object."""

        return (v.graph for v in dict.values(self))

    def itergraphs(self):
        """Iterates over the STGraphs."""

        return (v.graph for v in dict.values(self))

    def graphs(self):
        """Return a list of the STGraphs."""

        return [v.graph for v in dict.values(self)]

    def itersplits(self):
        """Iterates over the GraphSplit objects instead of the STGraphs."""

        return dict.values(self)


class GraphSplit(object):
    """Holds the split information for a single graph after a distinction was
    used to split a set of graphs.

    :var graph: (:class:`~srpt.STGraph.STGraph`) the graph this split is about
    :var items: the items matching the distinction

    Note: *graph* and *items* are only available before pickling. After
    unpickling they are replaced by *graph_id* and *item_ids*. This avoids
    saving out the actual STGraph, STObject, STRelation, and STAttribtue objects
    to the pickle which would greatly increase their size. If the original split
    is needed it can be restored using :meth:`~.restoreGraph`.
    """

    def __init__(self, graph, items):
        """Create a GraphSplit object.

        :param graph: the graph for this split
        :type graph: :class:`~srpt.STGraph.STGraph`
        :param items: a list of items that matched the distinction
        """
        self.graph = graph
        self.graph_id = graph.id
        self.items = items if items is not None else []

    def restoreGraph(self, graphs):
        """Reconstructs the pickled GraphSplit by turning the list of
        graph and item id's back into the real objects. This of course
        requires that the correct graph is in *graphs*.

        :param graphs: the graphs to search through to find the graph this split
            refers to
        :type graphs: Dict[graph.id => :class:`~srpt.STGraph.STGraph`]
        """
        self.graph = graphs[self.graph_id]

        self.items = []
        for item_type, item_id in self.item_ids:
            if item_type == 'o':
                self.items.append(self.graph.objects[item_id])
            elif item_type == 'r':
                self.items.append(self.graph.relations[item_id])
        self.item_ids = None

    def __getattr__(self, name):
        """Allows an GraphSplit object to act as a graph object by proxying
        attribute accessing to *self.graph*.
        """

        try:
            return getattr(self.graph, name)
        except:
            raise(AttributeError, name)


class Split(object):
    """Holds all the information from a distinction when it splits the data as two
    :class:`GraphSplitsDict`, yes and no, mapping graph.id's to GraphSplit objects.
    """

    def __init__(self, distinction):
        """Create an empty split object for the *distinction*.

        :param distinction: the distinction that created this split
        :type distinction: Subclass of :class:`~srpt.distinctions.Distinction`
        """

        self.distinction = distinction


        self.base_splits = []

        self.yes = GraphSplitsDict()
        self.num_yes_graphs = 0
        self.num_yes_items = 0

        self.no = GraphSplitsDict()
        self.num_no_graphs = 0
        self.num_no_items = 0




    def log_rank(self,class_labels,config):
        ###we will try to set the time row in the config object
        t=config.get('time_list', None)
        get_time = {t[i]: i for i in range(len(t))}
        N = len(t)
        y = np.zeros((3, N))
        d = np.zeros((3, N))
        count_sup = np.zeros((N, 1))
        count_inf = np.zeros((N, 1))
        for g in self.yes.itervalues():

            t_idx = get_time[g.flood_time]
            count_sup[t_idx] = count_sup[t_idx] + 1
            if g.class_label:
                d[2][t_idx] = d[2][t_idx] + 1
        for g in self.no.itervalues():
            t_idx = get_time[g.flood_time]
            count_inf[t_idx] = count_inf[t_idx] + 1
            if g.class_label:
                d[1][t_idx] = d[1][t_idx] + 1
        nb_inf = self.num_no_graphs
        nb_sup = self.num_yes_graphs
        for i in range(N):
            y[1][i] = nb_inf
            y[2][i] = nb_sup
            y[0][i] = y[1][i] + y[2][i]
            d[0][i] = d[1][i] + d[2][i]
            nb_inf = nb_inf - count_inf[i]
            nb_sup = nb_sup - count_sup[i]
        num = 0
        den = 0
        for i in range(N):
            if y[0][i] > 0:
                num = num + d[1][i] - y[1][i] * d[0][i] / float(y[0][i])
            if y[0][i] > 1:
                den = den + (y[1][i] / float(y[0][i])) * y[2][i] * ((y[0][i] - d[0][i]) / (y[0][i] - 1)) * d[0][i]

        if (den==0):
            L=0
        else:
            L = num / math.sqrt(den)
        self.rank=abs(L)
        return abs(L)


    """

    def calcContingencyTable(self,class_labels,config, stat=None):
        Calcualtes a contingency table on this split by looking at the distribution
        of classes that went down the yes and no branches.

        :param config: the Config object that contains all the experiment info
        :type config: :class:`~utilities.Config.Config`
        :param stat: the statistic used for determining the quality of a split,
            if not given (or None) then use the default from the above *config*
            object

        :returns: a contingency table (also stored at *self.table*)
        :rtype: :class:`~utilities.ContingencyTable.ContingencyTable`
        

        #self.config = config

        # setup the contingency table
        class_table = dict(yes={}, no={})
        for label in class_labels:

            class_table['yes'][label] = 0
            class_table['no'][label] = 0


            # count up the yes graphs
        for g in self.yes.itervalues():
            index = str(g.class_label)
            class_table['yes'][index] += 1


        # count up the no graphs
        for g in self.no.itervalues():
            index=str(g.class_label)
            class_table['no'][index] += 1


        self.class_table = class_table
        ########
        # pass the stat down to the contingency table
        if stat is None:
            stat = config.get('split_stat', None)

        if stat is not None:
            kwargs = {'stat': stat}
        else:
            kwargs = {}
        self.table = ContingencyTable(class_table, **kwargs)
        self.fit = self.table.fit
        self.pvalue = self.table.pvalue

        return self.table
    """
    def addYes(self, graph, matched_items):
        """Adds a graph and its matching items to the *yes* set of graphs.

        :param graph: the graph the items belong to
        :type graph: :class:`~srpt.STGraph.STGraph`
        :param matched_items: a list of items that matched the distinction
        """

        g = GraphSplit(graph, matched_items)
        g.split = self

        self.yes[graph.id] = g
        self.num_yes_graphs += 1
        self.num_yes_items += len(matched_items)

    def addNo(self, graph, matched_items):
        """Adds a graph and its matching items to the *no* set of graphs.

        :param graph: the graph the items belong to
        :type graph: :class:`~srpt.STGraph.STGraph`
        :param matched_items: a list of items that matched the distinction
        """

        g = GraphSplit(graph, matched_items)
        g.split = self

        self.no[graph.id] = g
        self.num_no_graphs += 1
        self.num_no_items += len(matched_items)

    def addBaseSplit(self, base_split):
        """Adds a split form a base distinction for when the distinction that
        created this distinction is a conjugate distinction.

        :param base_split: the split from a base distinction
        :type base_split: :class:`.Split`
        """

        self.base_splits.append(base_split)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop('config', None)
        state.pop('distinction', None)
        state['yes'] = list(self.yes.keys())
        state['no'] = list(self.no.keys())
        return state

    def __str__(self):
        return repr(self)

    def toFullString(self, expand_items=False):
        """Really huge string of the split."""

        def formatGraphs(s, graphs, expand_items):
            for g in graphs.itersplits():
                s += '\t\t%r\n' % g.graph
                for o in g.items:
                    if expand_items:
                        s += '%s\n' % o.__str__(indent=1)
                    else:
                        s += '\t\t\t%r\n' % (o)
            return s

        s = 'Split[#yes=%s, #no=%s\n' % (len(self.yes), len(self.no))
        s += '\tYes Graphs:\n'
        s = formatGraphs(s, self.yes, expand_items)
        s += '\t\n'
        s += '\tNo Graphs:\n'
        s = formatGraphs(s, self.no, False)
        s += ']'
        return s

    def __repr__(self):
        s = 'Split[tot=%sg(%si) yes=%sg(%si), no=%sg(%si)' % (
            self.num_yes_graphs + self.num_no_graphs,
            self.num_yes_items + self.num_no_items,
            self.num_yes_graphs,
            self.num_yes_items,
            self.num_no_graphs,
            self.num_no_items)
        if hasattr(self, 'class_table'):

            s = '%s, dist=y:%s, n:%s' % (s, self.class_table['yes'], self.class_table['no'])
        if len(self.base_splits) > 0:
            base = map(repr, self.base_splits)
            s = '%s, base_splits=%s' % (s, base)

        return '%s]' % (s)