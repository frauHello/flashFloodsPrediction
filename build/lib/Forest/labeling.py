import pickle
from experiment.ExperimentLoader import loadExperimentFile
from utilities.Config import Config
from utilities.utils import splitDict
import pandas as pd

treefile=r"C:\Users\user\Desktop\GraduationProject\ST_survival_RF\experiments\flood\run=0_numtrees=2_samples=5_depth=5_distinctions=all_stat=logrank.pkl"
experiment_Path = r"C:\Users\user\Desktop\GraduationProject\ST_survival_RF\experiments\flood.exp"

t = []
for i in range(1, 19):
    t.append(i)
config = Config()
config.DEBUG = True
config['time_list']=t
config['load_graphs_from_xml']=True
loadExperimentFile(config, filename=experiment_Path, experiment_name="flood")

forests = pickle.load(open(treefile, 'rb'))
training_labeling = forests.labelGraphs(config.graphs, config.time_list)
print("training labeling:")
print(training_labeling)