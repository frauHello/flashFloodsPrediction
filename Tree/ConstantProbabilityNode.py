"""
The simplest leaf node for an SRPT that calculates the simple probability
distribution of the classes passed into this node during training. The
probability of class "c" is in the PDF is::

    num_graphs_of_class_c/total_number_of_graphs

If the tree's config object has "laplace" set to True then the probability
of class "c" is adjusted using the laplacian correction::

    (num_graphs_of_class_c + 1)/(total_number_of_graphs + number_of_classes)

For classification a graph is labeled as having the class label with the
highest probability in the PDF.
"""

from collections import defaultdict

from Tree.Node import Node
from lifelines import NelsonAalenFitter
import numpy as np
import matplotlib.pyplot as plt

from utilities.utils import np2py, cleandict


class ConstantProbabilityNode(Node):
    """
    :var max_class: the class_label that has the highest probability
    :var count: number of graphs used to determine the pdf during training
    :var frequency: (Dict[class_label => count]) number of graphs of each class
        that were used to make the PDF during training
    :var pdf: (Dict[class_label => probability]) probability of each class at
        this node, used during labeling
    :var depth: the depth of this node in the SRPT
    """

    # attributes to be pickled by instanceReducer
    _init_args = ( 't', 'total', 'depth','count')
    ####I have removed count from here to see if I will get the pickle error another time
    ##I have added count.


    def __init__(self, parent, graphs=None):
        """Create a new constant probability node that labels a graph based
        upon the PDF at this node.

        :param parent: the parent node of this node
        :param graphs: the graphs to calculate the PDF from
        :type graphs: Dict[graph.id=>STGraph]
        """
        Node.__init__(self, parent)
        """
        if graphs is not None:
            #self._calcPDF(graphs)
            self.compute_leaf(graphs)
        """

    def _initWithArgs(self, args):
        """Callback for unpickling with :func:`~utilities.utils.instanceLoader`."""

        self.t = args[0]
        self.total = args[1]
        self.depth = args[2]
        self.count = args[3]
        #self.depth = args[4]
        self.children = []

    def growTree(self, graphs, class_labels):
        """Estimate the probability distribution of samples that make it to this
        leaf node.

        :param graphs: the graphs to use for calculating the PDF at this node
        :type graphs: Dict[graph.id=>STGraph]
        """
        print("Inside growTree function of the ConstantProbabilityNode class")
        # self._calcPDF(graphs,class_labels)
        self.compute_leaf(graphs, class_labels)

        def format(d):
            return ', '.join(map(lambda kv: '%s=%s' % kv, d.iteritems()))


        if self.config.get('verbose', False):
            print(('|    ' * (self.depth)), 'ConstantProbability')
            print(('|    ' * (self.depth)), 'ConstantProbability [%s]' %self.count)
            print(('    ' * (self.depth)), '  [flood_times: %s, total_number_graphs: %s]' % (self.t, self.total))


    def compute_leaf(self, graphs, class_labels):
        count = {}
        #self.compute_chf(graphs)
        for sample in graphs.itervalues():
            for label in class_labels:
                count.setdefault((sample.flood_time, label), 0)

            count[(sample.flood_time, sample.class_label)] = count[(sample.flood_time, sample.class_label)] + 1

        t = list(set([c[0] for c in count]))
        t.sort()
        total = len(graphs)
        self.count = count
        self.t = t
        self.total = total



    def labelGraphs(self, graphs, labels,time_array):
        """
        The survival estimate associated with a terminal node is provided by the kaplan-Meier (KM) estimator.
        let t1<t2< .....tm be the distinct death time in the terminal node h, and let dk and Yk equal the number
        of deaths and individual at risk respectively at time tk in h.
        then, the KM estimator is S(t)=product(1-dk/Yk) where tk < t
        
        Labels the graphs and places the labeling into the dict labels.

        :param graphs: the graphs to label
        :type graphs: Dict[graph.id => STGraph]
        :param labels: the dict to place the labels of the graphs into, a label
            is a survival estimate associated by the curent terminal node to a graph
        :type labels: Dict[graph => label]
        """
        count = self.count
        t = self.t
        total = self.total

        d={el:(0,1) for el in time_array}
        for time_step in time_array:

            s = 1
            h = 0
            survivors = float(total)
            for ti in t:
                if ti <= time_step:

                    s = s * (1 - count[(ti, '1')] / survivors)
                    h=h + count[(ti, '1')] / survivors
                survivors = survivors - count[(ti, '1')] - count[(ti, '0')]
            d[time_step]=(h,s)
        for graph in graphs.values():
                labels[graph] =d



    """
    def labelGraphs(self, graphs, labels):
        self.chf = self.chf.cumulative_hazard_

        print(self.chf)
        print(self.chf.shape)
        for graph in graphs.values():
            h= self.chf.iloc[ graph.flood_time,]
            labels[graph] = h

        return labels




    def compute_leaf(self,graphs,class_labels):
        t=[]
        e=[]
        self.chf=NelsonAalenFitter()

        for sample in graphs.itervalues():
            t.append(float(sample.flood_time))
            e.append(float(sample.class_label))

        time=np.asarray(t)
        event=np.asarray(e)
        self.chf.fit(time, event_observed=event, timeline=range(78,145))
        self.chf.cumulative_hazard_.plot()
        plt.show()

    """
####we can use this function and lifeline library to visualize for ezh tree the chf and the survival function
    def compute_chf(self,graphs):
        t = []
        e = []
        self.chf = NelsonAalenFitter()

        for sample in graphs.itervalues():
            t.append(float(sample.flood_time))
            e.append(float(sample.class_label))

        time = np.asarray(t)
        event = np.asarray(e)
        print("time dtype: {0}".format(time.dtype))
        print("event dtype: {0}".format(event.dtype))
        self.chf.fit(time, event_observed=event, timeline=range(78, 145))
        print("cumulative hazard function:")
        print(self.chf.cumulative_hazard_)
        """
        print("confidence interval cumulative hazard function:")
        print(self.chf.confidence_interval_cumulative_hazard_)
        print("confidence interval:")
        print(self.chf.confidence_interval_)
        """



    def __str__(self, depth=0, text='', simple=False):
        #pdf = ', '.join(('%s=%0.4G(%s)' % (k, self.pdf[k], self.frequency[k]) for k in self.pdf.keys()))
        #lines = ['label: %s' % self.max_class, 'pdf: %s' % pdf]

        #return Node.__str__(self, depth, lines, simple=simple)
        pass
