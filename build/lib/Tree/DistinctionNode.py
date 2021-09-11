from .Node import Node
from utilities.utils import SimpleContainer
from time import time
from utilities.ContigencyTable import ContingencyTable


class DistinctionNode(Node):
    """
    An internal node of the SRPT tree that holds a distinction and has two children, one child/branch
    will get passed the graphs that match the nodes distinction, the other branch those graphs that dont.

    :var distinction: (:class:srpt.distinctions.Distinction) a distinction object
        that is used to split graphs
    :var num_samples: the number of distinctions sampled when picking the best
        distinction during training
    :var pval_threshold: the pvalue threshold useding during training
    :var stat: the statistic used for measuring the quality of distinction
        when sampling them during training
    :var depth: the depth of this node in the SRPT
    """

    # attributes to be pickled by instanceReducer
    _init_args = ('distinction', 'num_samples','children', 'depth', 'stat', 'split')

    def __init__(self, parent, distinction=None, graphs=None):
        """
        :param parent: the parent node of this node
        :param distinction: optional the distinction this node holds
        :param graphs: optional dict of graphs) to split
            using *distinction* and store the split with this node
        :type graphs: Dict[graph.id=>STGraph]
        """
        Node.__init__(self, parent)

        self.num_samples = self.config.get('num_samples', 0)
        self.split = None
        self.distinction = distinction

        # if a distinction and graphs are given, we can go ahead and split
        # the graphs and save the split
        if distinction is not None and graphs is not None:
            self.split = distinction.splitGraphs(graphs)

        # must have a split stat so if its not given, default to chi-squared
        self.stat = self.config.get('split_stat', 'logrank')
        self.min_graphs = ContingencyTable.getMinObservations(self.stat)

    def _initWithArgs(self, args):
        """Callback for unpickling with :func:`~utilities.utils.instanceLoader`."""

        self.distinction = args[0]
        self.num_samples = args[1]
        self.children = args[2]
        self.depth = args[3]
        self.stat = args[4]

        self.split = None if len(args) < 6 else args[5]
        for child in self.children:
            child.parent = self

    def growTree(self, graphs,class_labels):
        """Given a dict of graphs  attempt to grow this node by sampling random
        distinctions finding the best one that meets the pvalue threshold. 
        If a valid distinction is found return this node itself, if no valid 
        distinction is found return None.

        :param graphs: the graphs to use when determining the quality of a distinction
        :type graphs: Dict[graph.id=>STGraph]

        :returns: self if a suitable distinction is found. otherwise, None.
        """

        if len(graphs) < self.min_graphs:
            if self.config.get('verbose', False):
                # not enough graphs for fit statistic to be stable
                print('|    ' * self.depth, 'Breaking due to # graphs < %s' % self.min_graphs)
            return None

        # check to see if there is more than one class label in the graphs
        labels = set()
        for graph in graphs.values():
            if graph.class_label not in labels:
                labels.add(graph.class_label)
            if len(labels) > 1:
                break
        else:
            if self.config.get('verbose', False):
                # only a single class of graphs in the so no point in splitting
                print('|    ' * self.depth, 'Breaking due to single class: %s' % labels)
            return None

        # sample the distribution of distinctions by making a random distinction
        # and then testing its quality, the best one is picked in the end
        gen = self.config.distinction_generator
        best = SimpleContainer(d=None, split=None)



        for _ in range(self.num_samples):
            d = gen.getRandomDistinction(graphs)
            start = time()
            split = d.splitGraphs(graphs)
            end = time()
            gen.runtimes[d.__class__.__name__] += (end - start)
            #choose the split with the highest log rank
            split.log_rank(class_labels,self.config)

            if best.split is None or split.rank > best.split.rank:
                    best.d = d
                    best.split = split

        if best.d is not None:
            # a valid distinction was found that splits the data, so
            # grow the two branches of the tree
            self.distinction = best.d
            self.split = best.split
            self.split.distinction = self.distinction


            if self.config.get('verbose', True):
                print(('|    ' * (self.depth - 1)) + '|  ', self.distinction)
                #print(('|    ' * self.depth) + '| Labels:', ', '.join(best.split.distinction.col_names))
                print(('|    ' * self.depth) + '|    Yes:{0}'.format(best.split.num_yes_graphs))
                print(('|    ' * self.depth) + '|     No:{0}'.format(best.split.num_no_graphs))
                print(('|    ' * self.depth) + '|    log rank: %s' % (best.split.rank))

            # Node.growTree() knows how to grow a single node, either branching or
            # terminating at a leaf, so we just append what it returns
            if self.config.get('verbose', True): print(('|    ' * (self.depth)) + '+--->== YES ==')
            self.children.append(Node(self).growTree(self.split.yes,class_labels))
            if self.config.get('verbose', True): print(('|    ' * (self.depth)) + '|--->== NO ==')
            self.children.append(Node(self).growTree(self.split.no,class_labels))
            if self.config.get('verbose', True): print(('|    ' * (self.depth)) + '`')
            return self
        else:
            # no valid distinction found, indicate this to the calling function
            # by returning None
            if self.config.get('verbose', True):
                print('|    ' * self.depth, 'No significant split found')
            return None

    def labelGraphs(self, graphs, labels,time_step):
        """Label each graph with its predicited class label.

        :param graphs: the graphs to label
        :type graphs: Dict[graph.id=>STGraph]
        :param labels: the dict to place the tree's predicted label into
        :type labels: Dict[STGraph=>class_label]

        :returns: the dict passed in as *labels*
        """

        if len(graphs) == 0: return

        # use this nodes distinction to split the graphs
        split = self.distinction.splitGraphs(graphs)

        # Sanity check: if a distinction doesn't return all the graphs in one branch
        # of the split or the other, then something is very wrong!
        missing_ids = []
        for graph_id in graphs:
            if graph_id not in split.yes and graph_id not in split.no:
                missing_ids.append(graph_id)

        if len(missing_ids) > 0:
            raise Exception('Graph ids %s are missing after split %s' % (missing_ids, self.distinction))

        # send the matching graphs down the yes-branch
        self.children[0].labelGraphs(split.yes, labels,time_step)

        # send the non-matching graphs down the no-branch
        self.children[1].labelGraphs(split.no, labels,time_step)

        return labels

    def getProbabilities(self, graphs, probs=None):
        """Get the PDF for each graph.

        :param graphs: the graphs to get the PDF for
        :type graphs: Dict[graph.id=>STGraph] 
        :param probs: the dictionary to place the PDFs into
        :type probs: Dict[STGraph=>Dict[class_label=>probability]]

        :returns: the dict passed in as *probs*
        """
        if probs is None: probs = {}

        # use our distinction to split the graphs
        split = self.distinction.splitGraphs(graphs)

        # send the matching graphs down the yes-branch
        self.children[0].getProbabilities(split.yes, probs)

        # send the non-matching graphs down the no-branch
        self.children[1].getProbabilities(split.no, probs)

        return probs

    def __str__(self, depth=0, text='', simple=False):
        txt = [str(self.distinction)]
        if hasattr(self, 'split') and self.split is not None:
            txt.append(str(self.split))

        return Node.__str__(self, depth, txt, simple=simple)