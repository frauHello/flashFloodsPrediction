from collections import defaultdict
import numpy as np
from utilities.utils import weightedChoice, dynamicImport

from Tree.Distinction import DistinctionTypes
import sys
import traceback
import os

#: Defines the different named groups of distinction types, its a dictionary of
#: group names to a tuple of matching types.
#:
#:    * base - Basic
#:    * nontemp - Basic or Conjugate
#:    * basecon - Basic or Conjugate or Temporal-Conjugate
#:    * nonspatial - Basic or Conjugate or Temporal or Temporal-Conjugate

DISTINCTION_TYPES = defaultdict(lambda: ((DistinctionTypes.Basic,),  # @UndefinedVariable
                                         (DistinctionTypes.Conjugate,),  # @UndefinedVariable
                                         (DistinctionTypes.Temporal,),  # @UndefinedVariable
                                         (DistinctionTypes.Spatial,),  # @UndefinedVariable
                                         (DistinctionTypes.Temporal, DistinctionTypes.Conjugate),  # @UndefinedVariable
                                         (DistinctionTypes.Spatial, DistinctionTypes.Conjugate),  # @UndefinedVariable
                                         ))

# setup the named groups of distinctions
DISTINCTION_TYPES["base"] = ((DistinctionTypes.Basic,),  # @UndefinedVariable
                             )
DISTINCTION_TYPES["nontemp"] = ((DistinctionTypes.Basic,),  # @UndefinedVariable
                                (DistinctionTypes.Conjugate,),  # @UndefinedVariable
                                )
DISTINCTION_TYPES["basecon"] = ((DistinctionTypes.Basic,),  # @UndefinedVariable
                                (DistinctionTypes.Conjugate,),  # @UndefinedVariable
                                (DistinctionTypes.Temporal, DistinctionTypes.Conjugate),  # @UndefinedVariable
                                )
DISTINCTION_TYPES["nonspatial"] = ((DistinctionTypes.Basic,),  # @UndefinedVariable
                                   (DistinctionTypes.Conjugate,),  # @UndefinedVariable
                                   (DistinctionTypes.Temporal,),  # @UndefinedVariable
                                   (DistinctionTypes.Temporal, DistinctionTypes.Conjugate),
                                   # @UndefinedVariable
                                   )

#: named list of distinctions
DISTINCTIONS_LIST = {}


#: the default list of distinctions that can be picked from, this needs to be
#: updated when a new distinction is added
DEFAULT_DISTINCTION_LIST = ( 'Attribute','TemporalPartialDerivative','TemporalAccumulate')




class DistinctionGenerator:
    """Generators distinctions for an SRPT given a subclass of distinctions"""

    def __init__(self, config, distinctions=None):
        """Create a distinction generator.

        :param config: used to get the *schema* and the command line provided
            distinctions (from *config.distinctions*). if *config.distinctions* is
            not set (or ``None``) then default to the parameter *distinctions*,
            this object is placed in *config.distinction_generator*
        :param distinctions: the list of distinctions this generator can use,
            if ``None`` then use :data:`.DEFAULT_DISTINCTION_LIST`
        """

        # grab some basic information

        self.config = config
        config.distinction_generator = self
        self.schema = config.schema

        # determine the types (conjugate, temporal, etc) of the distinctions
        distinction_types = config.get('distinctions', distinctions)


        # try to figure out what the user meant form the supplied argument
        # on the command line. check if it was a named group of types
        if distinction_types in DISTINCTION_TYPES:
            self.distinction_types = DISTINCTION_TYPES[distinction_types]
        # maybe it was a pythonic list of tuples off DistinctionTypes enums
        elif isinstance(distinctions, str):
            try:
                self.distinction_types = eval(distinctions, None, DistinctionTypes)
            except:
                self.distinction_types = DISTINCTION_TYPES['all']
        else:
            self.distinction_types = DISTINCTION_TYPES['all']

        # maybe it was a list of distinction names
        dist_list = self.config.get('distinctions', distinctions)
        if dist_list is None:
            dist_list = DEFAULT_DISTINCTION_LIST
        elif dist_list in DISTINCTIONS_LIST:
            dist_list = DISTINCTIONS_LIST[dist_list]
        else:
            if ',' in dist_list:
                dist_list = dist_list.split(',')
                if not all((dist in DEFAULT_DISTINCTION_LIST for dist in dist_list)):
                    dist_list = None
            else:
                dist_list = None

        # if all else fails, default to the full distinction list
        if dist_list is None or len(dist_list) == 0:
            dist_list = DEFAULT_DISTINCTION_LIST

        self.config.distinctions_list = dist_list

        self.counts = defaultdict(int)
        self.runtimes = defaultdict(float)
        self.distinctions = []

        # if it was a list of distinction names we need to import the modules
        # holding those distinctions
        for dis in self.config.distinctions_list:
            if type(dis) is str:
                name = '%sDistinction' % dis
                #print("We are trying to import the following distinction: %s"%name)
                self.distinctions.append(dynamicImport('Tree.%s' % name))
            else:
                self.distinctions.append(dis)

        # get the appropriate base distinctions, this selects the distinctions
        # that are not conjugate from the list of all distinctions
        self.base_types = []
        for dtype in self.distinction_types:
            if DistinctionTypes.Conjugate not in dtype:  # @UndefinedVariable
                self.base_types.append(dtype)

        # get the base distinctions and cdf
        self.base_distinctions, self.base_weights, self.base_cdf = self.getMatchingDistinctions(self.base_types,
                                                                                                valid_for_schema=False)

        # get all matching distinctions and the cdf

        self.all_distinctions, self.all_weights, self.all_cdf = self.getMatchingDistinctions(self.distinction_types,
                                                                                             valid_for_schema=False)
        ###we will get in all_weights:
        ####[9,1,1]which is the number of attributes associated with each distinction
        ####We will get as all_cdf: [0.81 0.90 1.]
        """
        print("self.all_distinctions:")
        print(self.all_distinctions)
        print("type(self.all_distinctions)")
        print(type(self.all_distinctions))
        print("self.all_weights")
        print(self.all_weights)
        print("self.all_cdf")
        print(self.all_cdf)
        """


        if config.get('verbose', False): print('Using distinctions:', [dist.__name__ for dist in self.all_distinctions])

    def getMatchingDistinctions(self, distinction_type, valid_for_schema=True):
        """Get a list of distinctions that match the given distinction type. 
        Optionally (by default) ensure that they are valid for the current schema.

        :param distinction_type: a list of tuples of DistinctionsTypes to match
            against all possible distinctions
        :param valid_for_schema: if ``True`` check each matching distinction by
            calling :meth:`srpt.distinctions.Distinction.isValidForSchema`

        :returns: a list of distinctions whose types match the given *distinction_type*
        """

        matching = []

        # if its not a list, ie the user passed in a single tuple, then convert
        # it into a list
        if type(distinction_type[0]) not in (list, tuple):
            distinction_type = (distinction_type,)

        # check each possible distinction to see if its type is in distinciton_types
        # if so, check (if needed) that its valid for the current schema
        for distinction in self.distinctions:
            match = False
            d_types = distinction.getType()
            if type(d_types[0]) not in (list, tuple):
                d_types = (d_types,)
            for d_type in d_types:
                if d_type in distinction_type:
                    match = True
                    break

            if match:
                #print("checking if {0} is valid for schema".format(d_type))
                if valid_for_schema:
                    if distinction.isValidForSchema(self.schema):
                        matching.append(distinction)
                else:
                    matching.append(distinction)

        # create a CDF for the matching distinctions by getting the number of
        # subtypes for each matching distinction. note we create a high and low
        # CDF, where high is closer to the true uniform, and low is a much rougher
        # estimate

        weights = np.zeros(len(matching), dtype=float)

        for i, distinction in enumerate(matching):

            weights[i] = distinction.getNumberOfSubtypes(self.config, True)
        cdf = weights.cumsum() / weights.sum()  # @UnusedVariable

        high_weights = np.zeros(len(matching), dtype=float)
        for i, distinction in enumerate(matching):
            high_weights[i] = distinction.getNumberOfSubtypes(self.config, False)
        high_cdf = high_weights.cumsum() / high_weights.sum()

        return matching, high_weights, high_cdf


    def getRandomDistinction(self, graphs, distinction_type=None):
        """Returns a random distinction from the list of all possible distinctions.
        Limits it to a subset of times if *distinction_type* is given.

        :param graphs:
        """

        # keep trying to pick a random distinction until successful
        distinction = None
        i = 0

        # error printing function that is useful when debugging distinctions
        def printerror():
            if self.config.get('veryverbose', False):
                etype, eval, tb = sys.exc_info()
                frames = traceback.extract_tb(tb)
                head = ''.join(traceback.format_exception_only(etype, eval))
                trace = ''
                for frame in frames[::-1]:
                    trace = '\n\t%s(%s) %s --> %s' % (os.path.basename(frame[0]), frame[1], frame[2], frame[3]) + trace
                    if 'srpt2' in frame[0]:
                        break

                print(head + trace)

        if distinction_type is None:
            #print("Inside DistinctionGenerator.getRandomDistinction")
            getDistinctionType = lambda: weightedChoice(self.all_distinctions, self.all_cdf)

        else:
            getDistinctionType = lambda: distinction_type

        while distinction is None:
            i += 1
            distinction_type = getDistinctionType()
            self.counts[distinction_type.__name__] += 1

            # conjugate distinctions need base distinctions to work
            if DistinctionTypes.Conjugate in distinction_type.getType():  # @UndefinedVariable
                num_needed = distinction_type.getNumberOfBaseDistinctionsNeeded()
                base_distinctions = []

                while len(base_distinctions) < num_needed:
                    base_distinction = None
                    while base_distinction is None:
                        base_distinction_type = weightedChoice(self.base_distinctions, self.base_cdf)
                        try:
                            base_distinction = base_distinction_type.getRandomDistinction(self.config, graphs)
                        except Exception:
                            base_distinction = None
                            printerror()
                    base_distinctions.append(base_distinction)

                # with the needed base distinctions get a random distinction
                try:
                    distinction = distinction_type.getRandomDistinction(self.config, graphs, *base_distinctions)
                except Exception:
                    distinction = None
                    printerror()
            else:
                try:
                    distinction = distinction_type.getRandomDistinction(self.config, graphs)
                except Exception:
                    distinction = None
                    printerror()
            """
            if i % 100 == 0:
                print(i)
            """
        return distinction