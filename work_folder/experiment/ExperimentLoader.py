import glob
import sys
import os
import pickle
import traceback
from io import StringIO

import argparse

from utilities.Config import Config
from XMLReader import GraphReader, SchemaReader
from utilities.utils import TimeIt


class ExperimentLoadingException(Exception): pass


class ExperimentLoadAction(argparse.Action):
    """Argparse action that loads a file specified on the command line into a config object."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Call back used by Argparse to the action. Loads an experiment file.

        :param parser: the argparse parser that is calling the action
        :param namespace: the namespace object which is modified by this action,
            in this case its a :class:`~utilities.Config.Config` object.
        :param values: the values passed in at the command line
        :param option_string: a string of options for the action. None are
            used by this action.
        """

        loadExperimentFile(namespace, values)


def loadExperimentFile(config, filename=None, experiment_name=None, load_schema=True, load_graphs=True):
    """Loads an experiment file which by default includes loading the schema and
    the graphs. This also sets up the paths loadSchema and loadGraphs will use
    should you choose to call them separately.

    :param config: config object to load the experiment file into, if
        ``None`` a new config object is created. it may also
        contain information about the experiment to load. if *config.exp* is set
        then it will be used to find the experiment file and the name of the
        experiment to load making *filename* and *experiment_name* not needed
    :param filename: optional path of the experiment file to load. if not given
        then *config.exp* is used to search for the correct file in the
        experiments directory
    :param experiment_name: which experiment to load from the experiment file. if
        not given then *config.exp* is used
    :param load_schema: if True, load the schema specified for the given
        experiment in the file
    :param load_graphs: if True, load the graphs specified for the given
        experiment in the file
    """

    if config is None:
        print("config is None")
        config = Config()



    #print("type(config): %s"%type(config))
    config.timer = config.get('timer', TimeIt())
    working_directory=config.get('w_dir','')

    if filename is None:
        #print("Inside file name is none, we are going to get the filename from inside config object")
        filename = config.get('exp', None)

    if filename is None:
        raise ValueError('Filename must be given or config.exp must be set to load the experiment file.')

    # attempt to find the experiment file by searching several potential
    # filenames and locations
    raw_filename = filename
    #print('raw_filename: %s'%raw_filename)
    potential_filenames = [raw_filename, '%s.exp' % raw_filename, working_directory+'/experiment/%s' % raw_filename,
                           working_directory+'/experiment/%s.exp' % raw_filename]
    filename = None
    for fname in potential_filenames:
        """
        print('fname: %s'%fname)
        print('os.path.exists(fname):%s'%os.path.exists(fname))
        print('os.path.isfile(fname)%s'%os.path.isfile(fname))
        """
        print(fname)
        if os.path.exists(fname) and os.path.isfile(fname):
            print("Inside the validated if")
            filename = fname
            break

    # the experiment file wasn't found, so maybe the user passed in a full
    # experiment name, which needs to be split into "experiment_group" (aka
    # experiment file without extension) and the experiment name
    if filename is None:
        parts = raw_filename.split('_')
        parts = [('_'.join(parts[0:i]), '_'.join(parts[i:])) for i in range(1, len(parts) + 1)]
        for part_file, exp_name in parts:
            fname = 'experiment/%s.exp' % part_file
            if os.path.exists(fname):
                filename = fname
                experiment_name = exp_name
                break

    if filename is None:
        sys.stderr.write('\nUnable to find experiment file using --exp=%s\n' % config.get('exp', ''))
        sys.exit(-1)

    if experiment_name == None:
        experiment_name = raw_filename
    config.experiment_name = experiment_name

    config.full_experiment_name = '%s_%s' % (os.path.splitext(os.path.basename(filename))[0], config.experiment_name)

    print('Loading experiment file from:', filename)

    config = Config.fromFile(filename=filename, config=config, groups_name='experiments', group_on='experiment_name',
                             group=experiment_name)

    # get the path to the experiment file, schema, pickles and xmls
    config.exp_file = filename
    config.exp_path = os.path.abspath(os.path.dirname(filename))

    def findPath(filename):
        # find the correct path, in the experiment file they are either
        # relative to the experiment file, or an absolute path
        path = os.path.dirname(filename)
        if path != os.path.abspath(path):
            return os.path.join(config.exp_path, path)
        else:
            return path

    if config.experiment_name is not None and config.experiment_name in config.experiments:
        # get all the correct paths for the schema, xml graphs, and pickle graphs
        print('Reading Experiment:', config.experiment_name)
        experiment = config
        experiment.schema_path = findPath(experiment.schema_file)
        experiment.schema_file = os.path.join(experiment.schema_path, os.path.basename(experiment.schema_file))

        experiment.pkl_path = findPath(experiment.pkl_graphs)
        experiment.pkl_file = os.path.join(experiment.pkl_path, os.path.basename(experiment.pkl_graphs))

        experiment.xml_path = findPath(experiment.xml_graphs)
        experiment.xml_file = os.path.join(experiment.xml_path, os.path.basename(experiment.xml_graphs))

        if load_schema:
            config.timer('read schema')
            loadSchema(experiment)
            config.timer('read schema')
            print('\t\tdone loading schema.')

        if load_graphs:
            config.timer('load graphs')
            loadGraphs(experiment)
            config.timer('load graphs')
            print('\t\tdone loading graphs.')
    else:
        print("Experiment Not Found:", config.experiment_name)
        print("Experiments In File: %s" % config.exp_file)
        print('\t%s' % '\n\t'.join(config.experiments.keys()))
        sys.exit()

    # clean up schema by removing ignored objects, relations, and attributes
    config.schema.removeIgnored()

    config.setdefault('distinctions', 'all')
    return config


def loadSchema(config):
    """Given a config that has had an experiment file loaded into it, load the schema file.

    :param config: the config object with a loaded experiment file and
        *schema_file* file set
    """
    print('\tReading schema from: %s' % (config.schema_file))
    config.schema = SchemaReader.loadFromXML(config.schema_file)


def loadGraphs(config):
    """Given a config that has had an experiment file loaded into it, load the
    graphs. This handles all the searching of the specified paths and attempts
    to resolve them to actual files. Should the graphs not be found in the
    pickle format then it will attempt to load them from the xml files. Once
    loaded, the graphs are placed in config.graphs.

    Loading XML files:

    - If *xml_path* is a single file, it is assumed to contain all the graphs
      and is loaded. The loaded graphs are pickled and then saved.
    - If *xml_path* is a directory all XML files are loaded as graphs and then
      saved as a pickle file
    - If *xml_path* contains characters for globbing '*' or '?' then the path
      is globbed on, all the returned files are loaded and then saved as a
      pickle file

    Saving the loaded XML as a pickle depends on if *pkl_path* is a file or a
    directory. If its a file name (ends in '.pkl'), then all the graphs are
    saved to a single file.

    If its a directory, then each graph is saved as a separate pickle. If
    *xml_path* is a filename, then each pickle file is named
    '%s_%s.pkl' %  (xml_path_without_extension, graph.id). Otherwise the pickle
    files are named '%s_%s.pkl' % (experiment_name, graph.id).

    Loading pickle files is done following the same logic as loading XML files.

    :param config: the config object with a loaded experiment file and *xml_path* or *pkl_path* set
    """
    # print config.__str__(pretty=True)

    errors = []
    graphs = {}
    from_pickle = False

    schema_filter = lambda s: ('schema' not in s.lower())

    # if the pkl's path exists we will attempt to load it from there
    if config.get('load_graphs_from_xml', False) == False and os.path.exists(config.pkl_path):
        # if its a file we can load it directly
        if os.path.isfile(config.pkl_file):
            pkl_files = [config.pkl_file]
        else:
            # if its a directory, we will glob for all the pickle files in that directory
            if os.path.isdir(config.pkl_file):
                pkl_files = glob.glob(os.path.join(config.pkl_file, '*.pkl'))

            # if its not a directory, perhaps it is a pattern, so we will glob
            # on itself
            else:
                # print 'Globbing:', pkl_file
                pkl_files = glob.glob(config.pkl_file)

                # Now that we've found some files, attempt to load them in
        config.pkl_files = pkl_files
        for file in pkl_files:
            try:
                print('\tReading graphs from: %s' % (file))
                g = pickle.load(open(file, 'rb'))
                print('\t\tread:', len(g))
                graphs.update(g)

            except:
                sys.stderr.write('\tFailed to load graphs from pickle file: %s\n' % (file))
                raise

        if len(graphs) > 0:
            from_pickle = True

    # if no graphs were loaded from the pickle files, attempt to load them from the XML
    if len(graphs) == 0:
        # if its a file, we can just use that
        if os.path.isfile(config.xml_file):
            xml_files = [config.xml_file]

        # if its a path with glob characters ('*', '?'), then glob on it
        elif '*' in config.xml_file or '?' in config.xml_file:
            print('globbing: %s' % config.xml_file)
            xml_files = filter(schema_filter, glob.glob(config.xml_file))

            # is it a directory?
        elif os.path.isdir(config.xml_file):
            xml_files = filter(schema_filter, glob.glob(os.path.join(config.xml_file, '*.xml')))

        else:
            if not os.path.exists(config.xml_file):
                print('\tGraphs file not found: %s' % (config.xml_file,))
            else:
                print('\tNo graphs files found using xml_graphs="%s"!' % (config.xml_graphs,))
            sys.exit()
        config.xml_files = xml_files
        print('\tnum files: %s' % len(xml_files))

        # we save separate pickle files per graph if the "config.pkl_file"
        # is a directory
        separate_pickles = os.path.isdir(config.pkl_file)
        if separate_pickles:
            pkl_path = config.pkl_file
            pkl_path = os.path.dirname(pkl_path)

            if os.path.isfile(config.xml_file):
                pkl_file, _ = os.path.splitext(os.path.basename(config.xml_file))

            else:
                pkl_file, _ = os.path.splitext(os.path.basename(config.exp_file))
                pkl_file = config.get('experiment_name', pkl_file)

            if not os.path.exists(pkl_path):
                os.makedirs(pkl_path)

            print('\t\tSaving graphs to folder as individual graphs:', pkl_path)

        # parse all the xml_files
        for file in xml_files:
            print( '\tReading graphs from: %s' % (file))

            # sometimes there are errors in the XML, but its really annoying
            # to stop loading all XML if one file is bad, so keep track
            # of the errors and show them to the user once all the files
            # have been "loaded"
            try:
                print("Hello, I am inside load from graph")
                g = GraphReader.loadFromXML(config.schema, file,keep_field_data=True)
                print('\t\tread', len(g))

            except (SystemExit, KeyboardInterrupt):
                # on sys.exit calls and keyboard interrupts really die, don't
                # keep going through the xml files
                traceback.print_exc(file=sys.stdout)
                sys.exit()

            # All other errors need to be caught and saved till later
            except:
                # create a stringio that we can write the traceback into
                tmpfile = StringIO()
                traceback.print_exc(file=tmpfile)

                # keep a list of the tracebacks
                err = '\n%s\n' % ('=' * 80)
                err += 'Error parsing: %s\n' % (file,)
                err += '\n%s\n' % tmpfile.getvalue()
                err += '\n%s\n' % ('=' * 80)
                errors.append((file, err))

                # alert user but go onto the next xml file
                sys.stderr.write(err)
                continue

            if separate_pickles:
                # save each graph separately
                for graph_id, graph in g.iteritems():
                    full_filename = os.path.join(pkl_path, pkl_file + '_%s.pkl' % graph_id)
                    print('\t\t\t', full_filename)
                    pickle.dump({graph_id: graph}, open(full_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # all graphs will go to a single file, so just add the graphs
                # to the list of all graphs
                graphs.update(g)

        # if graphs were loaded, save them to a pickle file
        if len(graphs) > 0 and not separate_pickles:
            path = os.path.dirname(config.pkl_file)
            pkl_file = config.pkl_file

            if not os.path.exists(path):
                os.makedirs(path)
            print('\t\tSaving graphs to pickle file:', pkl_file)
            pickle.dump(graphs, open(pkl_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # print "\t\tLoaded a total of %s graphs %s" % (len(graphs), ('from pkl' if from_pickle else 'from xml'))
    config.graphs = graphs

    # if there were errors loading XML files show all of them now to the user and
    # save them to a file the user can look at later.
    if len(errors):
        print('Error parsing these files:')
        fh = open('/tmp/xml_parse_errors.txt', 'w')
        for file, err in errors:
            print('\t', file)
            fh.write(err)
        print('Tracebacks saved to: /tmp/xml_parse_errors.txt')
        fh.close()
