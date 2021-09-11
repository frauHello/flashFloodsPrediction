"""
Represents the whole Spatio Temporal Relational Probability Tree from the root
node down. SRPTs are generally grown using :meth:`~srpt.SRPTree.SRPTree.growTree`.

The node types used by an SRPT are:
:class:`~srpt.DistinctionNode.DistinctionNode`for internal nodes 
:class:`~srpt.ConstantProbabilityNode.ConstantProbabilityNode`for leaf nodes. 
These can be changed by sub-classing the SRPTree class and
overriding :meth:`~srpt.SRPTree.SRPTree.getInternalNode` and
:meth:`~srpt.SRPTree.SRPTree.getLeafNode`.
"""

from time import time
from collections import defaultdict
from Tree.Node import Node
from Tree.STGraph import STGraph
from Tree.RandomDistinctionGenerator import DistinctionGenerator


class SRPTree(Node):
    """
    :var root: the root node of the tree

    These are taken from the Config object when the SRPTree object is created and
    are used for training/growing a tree:

    :var exerpiment_name: the name of the experiment used in training this tree
    :var run: the run this tree represents in the experiment
    :var num_samples: the number of distinctions sampled during training
    :var pvalue: the pvalue threshold used during training///this p_value is not needed anymore as the splitting 
    rule will be changed.
    :var max_depth: the maximum depth the tree could grow to during training
    :var split_stat: the statistic used for determining the quality/fit of a
        split when sampling distinctions during training----this will be log-rank

    These are only available after a tree has been trained using ``growTree()``:

    :var training_table: (:class:`utils.ContingencyTable.ContingencyTable`)
        the contingency table from classifying the graphs in the training set
    :var testing_table: (:class:`utils.ContingencyTable.ContingencyTable`)
        the contingency table from classifying the graphs in the testing set
    :var training_graph_ids: a list of the graph ids of the graphs in the training set
    :var testing_graph_ids: a list of the graph ids of the graphs in the testing set
    """

    #: These are the attributes of the tree that will get pickled automatically
    #: by :func:`~utilities.utils.instanceReducer`
    _init_args = ('root', 'depth', 'experiment_name', 'run', 'num_samples', 'max_depth', 'split_stat')


    def __init__(self, config):
        """Creates a new, empty (no nodes), SRPT ready for training.

        :param config: the :class:`~utilities.Config.Config` object that contains
            all the needed configuration information need for training. usable
            keys in config: full_experiment_name, num_samples, pvalue, run,
            max_depth, split_stat
        """

        # config contains all sorts of details for building the tree such as
        # number of samples to use, which distinctions we can choose from, etc
        #print("Inside SRPT TREE CONSTRUCTOR")
        self.config = config

        Node.__init__(self, self)

        # we want this information saved with the tree so that when its
        # unpickled we know how it will be trained
        self.experiment_name = config.get('full_experiment_name', None)
        self.num_samples = config.get('num_samples', None)
        self.run = config.get('run', None)
        self.max_depth = config.get('max_depth', None)
        self.split_stat = config.get('split_stat', 'logrank')



        # create the a distinction generator that can be used by
        # internal nodes for picking a random distinction
        DistinctionGenerator(config)

    def _initWithArgs(self, args):
        """Callback for unpickling with :func:`~utilities.utils.instanceLoader`."""

        args = iter(args)
        self.root = next(args)
        self.depth = next(args)
        self.experiment_name = next(args)
        self.run = next(args)
        self.num_samples = next(args)
        self.max_depth = next(args)
        self.split_stat=next(args)


        # old trees didn't have the tables pickled, but we still might want to
        # load them
        try:
            self.training_table = next(args)
            self.testing_table = next(args)
            self.training_graph_ids = next(args)
            self.testing_graph_ids = next(args)
            self.split_stat = next(args)
        except:
            pass

        self.children = [self.root]

    def setRoot(self, root):
        """Sets the root node of the tree. Used if you are building a tree
        by hand instead of growing it.

        :param root: the node to set as the root
        """

        self.root = root
        self.children.append(self.root)

    def growTree(self, graphs,class_labels):
        """Grow the SRPT using the given graphs as training instances.

        :param graphs: the graphs to use for training
        :type graphs: Dict[graph.id => :class:`srpt.STGraph.STGraph`]
        """

        start = time()

        # offset for tree being a node itself with a root node too, the logic
        # in the Node() class works off of the parents nodes depth, this makes
        # things work out correctly
        self.depth = 1
        if self.config.get('verbose', False): print('\nGrowing SRPT [')
        self.root = Node.growTree(self, graphs,class_labels)
        self.children.append(self.root)
        self.grow_time = (time() - start)
        if self.config.get('verbose', False): print(']\n')

    def labelGraphs(self, graphs,time):
        """Returns the predicted/classified labels of the graphs using this
        SRPT to do the classification.

        :param graphs: the graphs to label
        :type graphs: Dict[graph.id => :class:`srpt.STGraph.STGraph`]

        :returns: Dict[:class:`srpt.STGraph.STGraph` => class_label]
        """

        if isinstance(graphs, STGraph):
            graphs = {graphs.id: graphs}
        #print("Inside label graphs of SRPTree")
        #print("type(self.root)")
        #print(type(self.root))
        #print("self.root.depth: %s"%self.root.depth)
        labels = {}
        self.root.labelGraphs(graphs, labels,time)
        #print("labels:")
        #print(labels)
        return labels

    def getProbabilities(self, graphs):
        """Returns the PDF of class labels for each graph using this SRPT to
        generate the PDFs.

        :param graphs: the graphs to get the pdfs for
        :type graphs: Dict[graph.id => :class:`srpt.STGraph.STGraph`]

        :returns: Dict[:class:`srpt.STGraph.STGraph` => Dict[class_label=>probability]]
        """

        probs = {}
        self.root.getProbabilities(graphs, probs)
        return probs

    def __str__(self, depth=0,  text='',simple=False):
        if not simple:
            d = defaultdict(lambda: 'None')
            d.update(self.__dict__)
            txt = 'exp:%(experiment_name)s, run:%(run)s, samples:%(num_samples)s, maxdepth:%(max_depth)s, pvalue:%(pvalue)s, splitstat:%(split_stat)s' % d
            return Node.__str__(self, depth, txt)
        else:
            return 'SRPT[]'
