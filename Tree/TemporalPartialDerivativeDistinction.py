'''
For a Non-Field attribute:
    Is the partial derivative with respect to time on a non-field attribute *a*
    on a object or relation of type *t* > *v* over an extent of time *e*?

For a Field attribute:
    Is the partial derivative with respect to time of the magnitude of the
    [MIN, MAX, MEAN, STD] of the [CURL, DIVERGENCE, GRADIENT] of the
    field-attribute *a \geq v* over an extent of time *e*?

Example:

    * (Non-Field) TemporalPartialDerivative(Dt[car.speed] > 5, 10) - Is the partial
      derivative of the "car" objects "speed" attribute ever greater than "5" over
      the extent of "10" or less timesteps?

To do this the discrete partial derivative is calculated over the the attributes
values, *a*, for each extent [1...*e*]. For instance, ``a[i] - a[i-5]`` would be
the  discrete partial derivative over the extent of 5 for timestep *i*.

'''

from itertools import chain

import numpy as np

from utilities.utils import randomChoice, randomIter, np2py

from Tree.Distinction import Distinction, DistinctionTypes
from Tree.Split import Split


class TemporalPartialDerivativeDistinction(Distinction):
    _init_args = ('attrib', 'extent', 'split_value')

    def __init__(self, attrib,extent, split_value):
        self.attrib = attrib
        self.extent = extent
        self.split_value = np2py(split_value)


    def getReferencedTypes(self):
        return ((self.attrib.type,),)

    def splitGraphs(self, graphs):
        split = Split(self)
        no_data=-1000.0
        num_attribs = 0
        split_value = self.split_value if self.split_value >= 0 else -self.split_value

        # create a function for getting the correct attribute variables
        # this makes things a bit clearer

        valgetter = lambda attrib: attrib.value

        for graph in graphs.values():
            # get all the attributes of the correct type
            all_attribs = graph.attributes_by_type.get(self.attrib.id, None)


            if all_attribs is not None:
                matching = []
                not_matching = []
                num_attribs += len(all_attribs)

                # check each attribute
                for attrib in all_attribs:
                    vals = valgetter(attrib)
                    len_values = len(vals)
                    current_flood_time=attrib.parent.flood_time
                    max_extent = int(min(self.extent,current_flood_time-self.extent))

                    match = False
                    for idx in range(current_flood_time-max_extent):
                        # calculate the partial derivative for each extent
                        # this used to be done using numpy.convolve, but by hand
                        # ended up being easier to understand and get correct
                        # (and a tad bit faster)
                        if vals[idx] == no_data:
                             continue
                        for extent_step in range(1, max_extent):
                            j = idx + extent_step
                            if j >= len_values: continue
                            if vals[j] == no_data: continue

                            if (vals[j] - vals[idx]) >= split_value:
                                match = True
                                break
                        if match: break

                    if match:
                        matching.append(attrib.parent)
                    else:
                        not_matching.append(attrib.parent)
            else:
                matching = []
                not_matching = list(chain(graph.objects.values(), graph.relations.values()))

            if len(matching) > 0:
                split.addYes(graph, matching)
            else:
                split.addNo(graph, not_matching)
        return split

    @staticmethod
    def getNumberOfSubtypes(config, low_estimate=True):
        if low_estimate:
            return 1
        else:
            count = 0
            for attrib in config.schema.all_attributes.values():
                if ('Array' in attrib.type and 'LatLong' not in attrib.type
                    and 'Coord' not in attrib.type) or 'Field' in attrib.type:
                    count += 1
            return count

    @staticmethod
    def getType():
        return (DistinctionTypes.Temporal,)  # @UndefinedVariable

    @staticmethod
    def isValidForSchema(schema):
        for attrib in schema.all_attributes.values():
            if ('Array' in attrib.type and 'LatLong' not in attrib.type
                and 'Coord' not in attrib.type) or 'Field' in attrib.type:
                return True
        return False

    @staticmethod
    def getRandomDistinction(config, graphs):
        """Picks a random existence distinction"""

        # filter out invalid attributes, we only want attributes
        # that are of type int or float and array (hence temporal)
        no_data = -10000.0
        valid_attribs = []
        for attrib in config.schema.all_attributes.values():
            if ('Array' in attrib.type and 'LatLong' not in attrib.type
                and 'Coord' not in attrib.type) or 'Field' in attrib.type:
                valid_attribs.append(attrib)

        if len(valid_attribs) == 0:
            return None

        # pick a random attribute from the valid ones
        attrib = randomChoice(valid_attribs)

        stat = (None, None)
        # we need a split value, so run through the graphs
        real_attrib = None
        for graph in randomIter(graphs):
            # get attributes of the correct type from this graph
            all_attribs = graph.attributes_by_type.get(attrib.id, None)
            if all_attribs is None:
                continue

            all_attribs = [attrib for attrib in all_attribs if
                           attrib.parent.end_time - 1 > attrib.parent.start_time]

            if len(all_attribs) == 0:
                continue

            # pick a random attrib and get the value of the stat
            for attrib in randomIter(all_attribs):
                real_attrib = attrib
                values = real_attrib.value
                flood_time=real_attrib.parent.flood_time

                # we need at least three values to calculate a partial derivative
                # and an extent
                if len(values) < 3:
                    real_attrib = None

                if real_attrib is not None:
                    break
            if real_attrib is not None:
                break

        if real_attrib is None or len(values) - 1 < 3:
            return None

        # pick the extent to calculate the derivative over
        extent = np.random.randint(1,flood_time - 1)

        # pick the split value now that we have an attribute
        idx = np.random.randint(0, flood_time - extent)
        while(values[idx]==no_data):
            idx = np.random.randint(0, flood_time - extent)
        while(values[idx+extent]==no_data):
            extent = np.random.randint(1, flood_time - 1)

        split_value = values[idx + extent] - values[idx]
        #print("this is a temporal derivative split with split_value={0}".format(split_value))
        dist = TemporalPartialDerivativeDistinction(attrib, extent, split_value)

        return dist

    def getSplitType(self):
        return ('Temporal Partial Derivative',)

    def __str__(self):
        s = 'TemporalPartialDerivativeDistinction[attrib=%(attrib)r %%s, extent=%(extent)s, split=%(split_value)0.2F]' % self.__dict__
        return s