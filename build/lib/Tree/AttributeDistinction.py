'''
For a Temporal-Scalar attribute:
    Does an temporal-scalar attribute *a* have a [MIN, MAX, MEAN, STD] value *> v*?

For a Discrete attribute:
    Does an discrete attribute *a* have a value *= v*?

Examples:

    * (Discrete) Attribute(car.color = 'blue') - Is there a "car" object whose
      "color" attribute is "blue"? I.e Is there a blue car?
    * (Temporal-Scalar) Attribute(Max(car.speed) >= 75) - Is there a "car" object
      whose "speed" attribute has had a "maximum" value >= "75"?

'''
from itertools import chain

from utilities.utils import randomChoice, randomIter, np2py
from Tree.Distinction import Distinction, DistinctionTypes
from Tree.Split import Split
from Tree.STAttribute import AttributeStats, DataTypes
###AttributeStats = Enum('Stats', 'Mean', 'Median', 'Std', 'Mode', 'Exact', 'LessThan', 'Max', 'Min')

class AttributeDistinction(Distinction):
    _init_args = ('attrib', 'stat', 'split_value')

    def __init__(self, attribute_type, stat, split_value):
        self.attrib = attribute_type
        self.stat = stat
        self.split_value = np2py(split_value)

        self.setupSplitGraphs()

    def getReferencedTypes(self):
        return (self.attrib,)

    def setupSplitGraphs(self):
        if self.stat == AttributeStats.Mode:
            splitCompare = lambda self, attrib: attrib.stats[self.stat] == self.split_value
        elif self.stat == AttributeStats.Exact:  # @UndefinedVariable
            splitCompare = lambda self, attrib: attrib.value == self.split_value
        elif self.stat == AttributeStats.LessThan:  # @UndefinedVariable
            splitCompare = lambda self, attrib: attrib.value < self.split_value
        else:
            splitCompare = lambda self, attrib: attrib.stats[self.stat] >= self.split_value

        splitCompare.__name__ = 'splitCompare'
        self.splitCompare = splitCompare.__get__(self, AttributeDistinction)

    def splitGraphs(self, graphs):

        split = Split(self)

        for graph in graphs.values():
            all_attribs = graph.attributes_by_type.get(self.attrib, None)

            if all_attribs is not None:
                matching = []
                not_matching = []

                if self.splitCompare(all_attribs[0]):
                        #print("attrib value {0}".format(all_attribs[0]))
                        matching.append(all_attribs[0].parent)
                else:
                        not_matching.append(all_attribs[0].parent)
            else:
                matching = []
                not_matching = list(chain(graph.objects.values(), graph.relations.values()))

            if len(matching) > 0:
                split.addYes(graph, matching)
            else:
                split.addNo(graph, not_matching)

        return split

    @staticmethod
    def isValidForSchema(schema):
        for attrib in schema.all_attributes.values():
            if attrib.type in (DataTypes.Coord, DataTypes.LatLong):  # @UndefinedVariable
                pass
            elif attrib.type in (DataTypes.CoordArray, DataTypes.LatLongArray):  # @UndefinedVariable
                pass
            else:
                return True
        return False

    @staticmethod
    def getNumberOfSubtypes(config, low_estimate=True):
        if low_estimate:
            return 1
        else:
            count = 0
            for attrib in config.schema.all_attributes.values():
                if attrib.type in (DataTypes.Coord, DataTypes.LatLong):  # @UndefinedVariable
                    pass
                elif attrib.type in (DataTypes.CoordArray, DataTypes.LatLongArray):  # @UndefinedVariable
                    pass
                else:
                    count += 1
            return count

    @staticmethod
    def getType():
        return (DistinctionTypes.Basic,)  # @UndefinedVariable

    @staticmethod
    def getRandomDistinction(config, graphs,*base_distinctions):
        """Picks a random existence distinction"""

        # filter out invalid attributes, we only want attributes
        # that are of type int or float
        def attribFilter(attrib):
            if attrib.type in (DataTypes.Coord, DataTypes.LatLong):  # @UndefinedVariable
                return False
            elif attrib.type in (DataTypes.CoordArray, DataTypes.LatLongArray):  # @UndefinedVariable
                return False
            else:
                return True

        #valid_attribs = filter(attribFilter, config.schema.all_attributes.values())
        valid_attribs = [item for item in config.schema.all_attributes.values() if attribFilter(item)]

        # pick a random attribute from the valid ones
        attrib = randomChoice(valid_attribs)

        # pick a random stat to split on
        if attrib.type in (DataTypes.IntArray, DataTypes.FloatArray,
                           DataTypes.DiscreteArray, DataTypes.IntField,  # @UndefinedVariable
                           DataTypes.FloatField):  # @UndefinedVariable
            stat = AttributeStats.pickRandom([AttributeStats.Exact, AttributeStats.LessThan])  # @UndefinedVariable
        else:
            stat =AttributeStats.LessThan # @UndefinedVariable

        # we need a split value, so run through the graphs
        real_attrib = None
        for graph in randomIter(graphs):
            # get attributes of the correct type from this graph
            all_attribs = graph.attributes_by_type.get(attrib.id, None)

            # if the are matching attribs add the to the list
            if all_attribs is not None:
                # pick a random attrib and get the value of the stat
                real_attrib = randomChoice(all_attribs)
                break
        if real_attrib is None:
            return None

        if stat not in (AttributeStats.Exact, AttributeStats.LessThan):  # @UndefinedVariable
            split_value = real_attrib.stats[stat]
        else:
            split_value = real_attrib.value
        #print("this is an attribute split, with value={0}, paramter={1} and stat={2}".format(split_value,attrib.id,stat))
        dist = AttributeDistinction(attrib.id, stat, split_value)

        return dist

    def getSplitType(self):
        return ('Attribute', AttributeStats.rawName(self.stat))

    def __str__(self):
        return 'AttributeDistinction[attrib=%(attrib)s, stat=%(stat)s, split=%(split_value)s]' % self.__dict__