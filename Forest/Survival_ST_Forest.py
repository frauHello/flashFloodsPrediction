"""A Spatiotemporal Relational Random Forest (SRRF) is a random forest of 
:class:`SRPTs <srpt.SRPTree.SPRTree>`. SRRFs are generally grown using 
:meth:`~.SRRForest.growForest`. However, they can be manually constructed if needed. 
"""

from time import time, ctime
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
from Tree.SRPTree import SRPTree
from utilities.utils import dictFromKeys, maxKey
from utilities.concordanceIndex import concordance_index
import pandas as pd
from utilities.utils import danger_level



class SRRForest:
    """This class represents a SRRF made of SRPTs, and contains the logic
    needed to grow a SRRF.

    :var config: the :class:`~utilities.Config.Config` object holding all the 
        experiment info and parameters governing the construction of the SRRF
        and its component SRPTs
    :var num_trees: the number of trees to grow in the forest
    :var join_pdfs: determines how the PDFs of the trees should be combined into
        a single PDF for the forest
    :var trees: (List[:class:`~sprt.SRPTree.SRPTree`]) the trees of the forest

    The following attributes only exist after growTree:

    :var in_bag_graph_ids: a list of sets of graph ids of the in-bag graphs used
        for growing a SRPT. these are a bootstrap resampling of the graphs used 
        during training. *self.in_bag_graph_ids[i]* were used to train *self.trees[i]*
    :var out_of_bag_graph_ids: a list of sets of graph ids of the out-of-bag graphs used
        for growing a SRPT. these are the graphs *not* picked for the in-bag set.
        *self.out_of_bag_graph_ids[i]* were *not* used to train *self.trees[i]*
    """

    def __init__(self, config):
        """Constructs an empty SRRF ready for growing or manual construction.

        :param config: the :class:`~utilities.Config.Config` object holding all the 
            experiment info and parameters governing the construction of the SRRF
            and its component SRPTs.

        The following keys are needed by the SRRF: 

            * num_trees - the number of trees to grow in the forest
            * join_pdfs - how the pdfs from the individual trees should be joined
              to create a single PDF for the forest. Possible values are: *voting*
              where each tree gets a single vote, being its highest probability 
              class, and the votes are used to create a PDF; *mean* takes the
              mean PDF of all the tree PDFs to create the single forest PDF, the
              default is 'voting'

        """
        self.config = config
        self.num_trees = config.num_trees
        self.trees = None
        self.class_labels = []
        for label in config.schema.class_labels:
            self.class_labels.append(label)

    def growForest(self, graphs):
        """Using the *graphs* grow an SRRF with *self.num_trees* trees in it. The
        graphs a resampled using replacement (bootstrap resampling) to create an
        in-bag set of graphs for each tree, this also creates an out-of-bag set
        (the graphs not in the in-bag set) for each tree. The in-bag graphs are
        used for training each tree.

        After being grown each tree is evaluated on the in-bag and out-of-bag
        graphs 

        :param graphs: the graphs to use for training
        :type graphs: Dict[graph.id => :class:`~srpt.STGraph.STGraph`] 
        """


        # divvy up the graphs for oob error-estimation as per:
        #    http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr
        graph_ids = list(graphs.keys())
        num_graphs = len(graph_ids)

        all_idxs = np.arange(0, num_graphs)

        self.in_bag_graph_ids = []
        self.out_of_bag_graph_ids = []

        for tree_idx in range(self.num_trees):
            # take a bootstrap sample of the graph_ids
            in_bag_idxs = np.random.randint(0, num_graphs, size=num_graphs)
            in_bag_graph_ids = np.take(graph_ids, in_bag_idxs).tolist()
            in_bag_graph_ids.sort()

            # the out of bag indices are those not chosen for the in bag
            out_of_bag_idxs = np.setdiff1d(all_idxs, np.unique(in_bag_idxs))
            out_of_bag_graph_ids = np.take(graph_ids, out_of_bag_idxs).tolist()
            out_of_bag_graph_ids.sort()

            self.in_bag_graph_ids.append(in_bag_graph_ids)
            self.out_of_bag_graph_ids.append(out_of_bag_graph_ids)

        # Grow the trees
        self.trees = []
        for tree_idx in range(self.num_trees):
            if self.config.get('verbose', False):
                print("Tree %d/%d (started %s)" % (tree_idx + 1, self.num_trees, ctime()))

            # get the actual trees from the in-bag ids and the out-of-bag ids
            in_bag_ids = self.in_bag_graph_ids[tree_idx]
            in_bag = dictFromKeys(graphs, in_bag_ids)

            out_of_bag_ids = self.out_of_bag_graph_ids[tree_idx]
            out_of_bag = dictFromKeys(graphs, out_of_bag_ids)

            tree = SRPTree(self.config)
            tree.training_graph_ids = in_bag_ids
            tree.growTree(in_bag,self.class_labels)

            # evaluate the tree on the in-bag and out-of-bag graphs

            in_bag_labeling = tree.labelGraphs(in_bag,self.config.time_list)
            out_bag_labeling = tree.labelGraphs(out_of_bag,self.config.time_list)
            tree.out_of_bag_labels=out_bag_labeling


            self.trees.append(tree)



    def labelGraphs(self, graphs,time,outfile,vis):
        """Labels the graphs using the SRRF and returns a labeling. A labeling
        is a Dict[STGraph => label] where a label is the mean value of the compute 
        survival over all the trees in the forest.

        :param graphs: the graphs to label
        :type graphs: Dict[graph.id => :class:`~srpt.STGraph.STGraph`]

        :returns: labeling of the graphs
        :rtype: float
        """


        return self._labelGraphsByMeanPDF(graphs,time,outfile,vis)


    def _labelGraphsByMeanPDF(self, graphs,time,outfile,vis):
        """Helper function that labels the graphs using the mean-pdf method.

        :params graphs: graphs to label

        :returns: labeling of the graphs
        """
        id=0
        data = ET.Element('results')
        lst = lambda: list(range(0, 2))
        forest_labels=defaultdict(lambda :defaultdict(lst))
        if (vis):
            df = pd.DataFrame(columns=['lon', 'lat', 'survival_probability', 'time', 'danger'])

        # label all the graphs with each tree
        for tree in self.trees:
            labeling = tree.labelGraphs(graphs,time)
            #print("tree:{0}".format(tree))



            # for each graph, look at the labeling returned
            for graph, h in labeling.items():
                # for each labeling, sum up the probabilities of each class label

                for t,label in h.items():

                    forest_labels[graph][t][0]+=label[0]
                    forest_labels[graph][t][1] += label[1]




        for graph,h in forest_labels.items():

            lat = graph.graph.attributes_by_type.get(('cell', 'lat'))[0].value
            lon = graph.graph.attributes_by_type.get(('cell', 'lon'))[0].value
            cell = ET.SubElement(data, 'cell')
            cell.set("id", str(id))
            item_lat = ET.SubElement(cell, 'lat')
            item_lon = ET.SubElement(cell, 'lon')
            item_lat.text = str(lat)
            item_lon.text = str(lon)

            for t,label in h.items():
                item_prob = ET.SubElement(cell, 'survival_proba')
                item_prob.set('step',str(t))
                label[0]/=self.num_trees
                label[1] /= self.num_trees
                item_prob.text = str(label[1])
                if(vis):
                    danger = danger_level(label[1])
                    df = df.append(
                        {'lon': lon, 'lat': lat, 'survival_probability': label[1], 'time': t, 'danger': danger},
                        ignore_index=True)
            id += 1
        mydata = ET.tostring(data)
        myfile = open(outfile, "wb")
        myfile.write(mydata)
            #print(forest_labels[graph])
        if vis:
            return df



    def getOutOfBagLabels(self):
        """Returns the labeling of the out-of-bag graphs for each tree. This
        can only be done if *self.config.graphs* contains the graphs that are
        in the out-of-bag set for each tree.

        :returns: labeling for each tree of out-of-bag graphs
        :rtype: Dict[SRPTree => {STGraph => labeling}}.
        """

        labels = defaultdict(dict)
        for tree_idx, tree in enumerate(self.trees):
            if hasattr(tree, 'out_of_bag_labels'):
                labels[tree_idx] = tree.out_of_bag_labels
            else:
                out_of_bag = dictFromKeys(self.config.graphs, self.out_of_bag_graph_ids[tree_idx])
                labeling = tree.labelGraphs(out_of_bag,time)
                tree.out_of_bag_labels = labeling
                labels[tree_idx] = labeling

        return labels

    def compute_oob_ensembles(self,graphs,outOfBagLabels):
        """
        Compute OOB ensembles.
        :return: List of oob ensemble for each sample.
        """
        print("this is inside compute oob ensemble")
        oob_ensemble_chfs ={}
        for graph in graphs.values():
            denominator =0
            numerator = []
            for tree_idx in range(self.num_trees):

                for key,val in outOfBagLabels[tree_idx].items():
                    chf=[]

                    if (graph.id==key.graph.id):
                        for c in val.values():

                            chf.append(c[0])
                        if denominator==0:
                            numerator=np.array(chf)

                        else:numerator =numerator +np.array(chf)
                        denominator = denominator + 1

                        break
                if denominator == 0:
                    continue
                else:

                    ensemble_chf = numerator/denominator
                    oob_ensemble_chfs[graph] = ensemble_chf


        return oob_ensemble_chfs





    def compute_oob_score(self, graphs, OutOfBagLabels):
        """
        Compute the oob score (concordance-index).
        :return: c-index of oob samples
        """
        oob_ensembles = self.compute_oob_ensembles(graphs, OutOfBagLabels)

        c = concordance_index(oob_ensembles)
        return c




    def __str__(self, **kwargs):
        return 'SRRF[numtrees=%s]' % (len(self.trees))

    def __len__(self):
        return self.num_trees