from collections import defaultdict
import argparse
import random
from utilities.Config import Config, ConfigLoadAction
from utilities.utils import splitDict, formatTime, argparser
from experiment.ExperimentLoader import loadExperimentFile
from Forest.Survival_ST_Forest import SRRForest
import pandas as pd


"""
defaults = dict(num_samples=5, max_depth=5, run=0, num_runs=1,stat='logrank',split_stat='logrank', 
num_folds=None,verbose=True, folds=None,load_graphs_from_xml=True,time_list=t)
"""
def _make_argparser():
    parser = argparse.ArgumentParser(
        prog='learn_srrf',
        description="This is the primary interface to growing SRRF's.")
    parser.add_argument('--distinctions', default='all')
    parser.add_argument('--exp',
                        help="Experiment configuration file.", required=True)
    parser.add_argument('--numsamples', dest='num_samples', type=int, default=100,
                        help="Number of samples of distinctions to make at each node of each tree while growing.")
    parser.add_argument('--numtrees', dest='num_trees', type=int, default=10,
                        help="Number of trees in the forest.")
    parser.add_argument('--time_list',
                       help="the time steps of the study")
    parser.add_argument('--maxdepth', dest='max_depth', type=int, default=5,
                        help="Maximum depth to grow each tree to.")
    parser.add_argument('--pvalue', type=float, default=.01,
                        help="P-Value threshold to be met while growing each tree.")
    parser.add_argument('--run', type=int, default=0,
                        help="Current run, or run to start at if --numruns > 1")
    parser.add_argument('--numruns', dest='num_runs', type=int, default=1,
                        help="Number of runs to do.")
    parser.add_argument('--splitstat', dest='split_stat', default='logrank',
                        help="The statistic used to determine the best split over num_samples")
    parser.add_argument('--defaults', action=ConfigLoadAction,
                        help="A config file that contains defaults.")
    parser.add_argument('--fromxml', dest='load_graphs_from_xml', action='store_true',
                        help="Force reload of graphs from XML")
    parser.add_argument('--savepath', dest='save_path',
                        help="Location to save the forests too")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Print all node information.")
    parser.add_argument('--underlabel', dest='underlabel', default='',
                        help="Name of class to be undersampled.")
    parser.add_argument('--underval', dest='underval', type=float, default=0.0,
                        help="Proportion of graphs to under sample from 0 to 1.")
    parser.add_argument('--opts', action='append', default=[])
    return parser


#@argparser(_make_argparser)
def main():
    """CLI for growing a SRRF."""
    t = []
    for i in range(1, 19):
        t.append(i)
    config = Config()
    config.DEBUG = True
    config['time_list']=t
    config['load_graphs_from_xml']=True

    defaults = dict(num_samples=100, max_depth=5, run=0, num_runs=1,num_trees=100, stat='logrank', split_stat='logrank', num_folds=None,exp='flood',
                    verbose=True, folds=None, load_graphs_from_xml=True, time_list=t)
    for key, value in defaults.items():
        cur_value = config.get(key, None)
        # print("key={0}:cur_value={1}".format(key,cur_value))
        config[key] = value if cur_value is None else cur_value
    config.DEBUG = True
    #loadExperimentFile(config, filename=experiment_Path, experiment_name="flood")
    #config.parseOpts()
    print('Start Grow Forest')
    growForest(config)


def growForest(config, load_exp_file=True):
    """Grows a SRRF given a config object that contains
    all the info needed to grow the forest. The command line arguments taken
    in :func:`.main` are what are needed in config.

    :param config: the config object with details on how to grow the forest
    :type config: :class:`~utilities.Config.Config`
    :param Boolean load_exp_file: If ``True`` load the experiment pointed to
        by *config.exp*, otherwise, just grow the tree by assuming everything
        is already in *config*.
    """

    silent = config.get('silent', False)
    experiment_Path = r"C:\Users\user\Desktop\Prediction_model\experiment\flood.exp"

    if load_exp_file:
        #loadExperimentFile(config, filename=config.exp)
        loadExperimentFile(config, filename=experiment_Path, experiment_name="flood")

    forests = []
    results = []


    # do multiple runs if needed. note that we start at config.run, not zero
    for run in range(config.num_runs):
        training_graphs, testing_graphs = splitDict(config.graphs, int(len(config.graphs) * .8), random=True)

        """
        # perform under-sampling if needed
        if hasattr(config, 'underlabel'):
            under_graphs = {}
            skip_count = 0
            for k in training_graphs.keys():
                if training_graphs[k].class_label == config.underlabel and random.random() <= config.underval:
                    skip_count += 1
                else:
                    under_graphs[k] = training_graphs[k]
            print('Undersampled ' + str(skip_count) + ' graphs')
            training_graphs = under_graphs
        """
        # print out some useful info on the class distribution
        counts = defaultdict(int)
        for graph in training_graphs.values():
            counts[graph.class_label] += 1
        print('training:', len(training_graphs), counts)

        counts = defaultdict(int)
        for graph in testing_graphs.values():
            counts[graph.class_label] += 1
        print('testing:', len(testing_graphs), counts)

        for graph in training_graphs.values():
            counts[graph.class_label] += 1
        print('total:', len(config.graphs), counts)

        print('\nrun:', run)
        config.run = run

        srrf = SRRForest(config)
        #srrf.growForest(training_graphs)
        srrf.growForest(config.graphs)
        forests.append(srrf)
        #srrf.training_graph_ids = list(training_graphs.keys())
        #training_labeling = srrf.labelGraphs(training_graphs,config.time_list)
        #outOfBagLabels=srrf.getOutOfBagLabels()
        #print("outOfBagLabels")
        #print(outOfBagLabels)
        #c=srrf.compute_oob_score(training_graphs, outOfBagLabels)
        #print("concordance index:")
        #print(c)
        config.saveTrees(srrf)

        #results.append(c)




        """

        df = pd.DataFrame(columns=['lon', 'lat', 'survival_probability', 'time'])


        srrf.testing_graph_ids = testing_graphs.keys()
        testing_labeling = srrf.labelGraphs(testing_graphs,config.time_list)







        for i,h in testing_labeling.items():

                lat = i.graph.attributes_by_type.get(('cell', 'lat'))[0].value
                lon = i.graph.attributes_by_type.get(('cell', 'lon'))[0].value
                for t, label in h.items():
                    df = df.append(
                    {'lon': lon, 'lat': lat, 'survival_probability': label[1], 'time': t},
                    ignore_index=True)

        sort_by_time = df.sort_values('time')
        print(sort_by_time.head())
        import plotly.express as px
        fig = px.scatter_mapbox(sort_by_time, lat="lat", lon="lon", hover_data=["survival_probability"],
                                color="survival_probability", animation_frame="time", animation_group="time",
                                color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10, height=500)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.show()
        """



        #config.saveTrees((srrf,)) ###config.saveTree is giving us an eror type error: unable to pickle dict keys.

    #print('numruns: %s' % (config.num_runs))
    #print(results)


    #return results


if __name__ == '__main__':
    main()
