

import numpy as np
import numpy.ma as maskedarray

from Tree.STObject import STObject

from utilities import Config

from utilities.Enum import Enum
"""from Prediction_model.utilities.Shapes import findShapes, getVolume"""
from utilities.utils import divergence, gradient, curl, instanceLoader
import utilities.utils as utils

#: An Enum of simple statistics applied to temporal non-field attributes
AttributeStats = Enum('Stats', 'Mean', 'Median', 'Std', 'Mode', 'Exact', 'LessThan', 'Max', 'Min')

#: An Enum of field functions that are applied to fields
FieldFuncs = Enum('FieldFuncs', 'Gradient', 'Curl', 'Divergence')

#: An Enum of statistics that are applied to fields
FieldAttributeStats = Enum('FieldAttributeStats', 'Mean', 'Std', 'Max', 'Min')

#: An Enum of the possible data-types for an attribute.
DataTypes = Enum('DataType', 'FloatVectorField', 'IntVectorField',
                 'FloatField', 'IntField',
                 'FloatArray', 'IntArray', 'DiscreteArray', 'CoordArray', 'LatLongArray',
                 'Float', 'Int', 'Discrete', 'Coord', 'LatLong', 'Str', 'StrArray')

# if the utilities.Config object is loaded before the STAttribute module,
# STAttribute can be forced to use regular dictionary based objects instead of slots
slotted = getattr(Config, 'slotted_stattribute', True)


class STAttribute(object):
    if slotted:
        __slots__ = ('id', 'name', 'type', 'parent', 'graph', 'value', 'mask', 'stats', 'field_stats',
                     'field_value', 'shapes', 'len', 'data_type', 'parent_type', 'parent_id', 'datalocation', 'corners',
                     'keep_field_data')

    def __init__(self, attr_id, name, parent, graph, data_type, value=None, mask=None):
        """A spatio-temporal attribute of an :class:`~srpt.STObject.STObject` or :class:`~srpt.STRelation.STRelation`.

        :param attr_id: the attributes unique id
        :param name: the name of the attribute
        :param parent: the object or relation this attribute belongs to
        :type parent: :class:`~srpt.STObject.STObject` or :class:`~srpt.STRelation.STRelation`
        :param graph: the STGraph this attribute (and its parent) belong to
        :param data_type: the data type of this attribute
        :type data_type: An element of :data:`~srpt.STAttribute.DataTypes`
        :param value: the value of this attribute
        :type value: numpy.array, str, int, or float
        :param mask: the mask of selecting which values in *value* are important.
            A boolean numpy array that has the same size of *value*. 
            mask[i,j] == True means value[i,j] is to be used 
        :type mask: Boolean numpy.array
        """

        # print attr_id, name, repr(parent), repr(graph), data_type, value.shape if value is not None else 'None', mask.shape if mask is not None else 'None'
        self.id = attr_id
        self.name = name
        self.parent = parent
        self.parent_id = None
        self.parent_type = None
        self.graph = graph
        self.value = value
        self.mask = mask
        self.stats = None
        self.field_stats = None
        self.field_value = None
        self.shapes = None
        self.len = 0


        if data_type not in DataTypes:
            raise ValueError('Invalid data_type [%s] must be one of: %s' % (data_type, DataTypes))
        else:
            self.data_type = data_type
        self.type = (self.parent.type, self.name)

        # add ourself to the main graph object
        if self.graph is not None:
            self.graph.attributes[self.id] = self
            self.graph.attributes_by_type[self.type].append(self)

        # add ourself to our parent
        self.parent.attributes[self.name] = self

        if self.value is not None:

            self._calcStats()

    def __reduce__(self):
        # instead of pickling potentially another copy of the parent (though
        # technically the memoization should only pickle a UID) we reduce
        # the parent down to an indicator of it being an object or a relation
        # and its id
        parent_type = self.parent_type
        if parent_type is None:
            parent_type = 'o' if isinstance(self.parent, STObject) else 'r'

        parent_id = self.parent_id
        if parent_id is None:
            parent_id = self.parent.id

        # full list of attributes saved in the pickle
        args = (self.id, self.name, parent_type, parent_id, self.data_type,
                self.mask, self.value, self.stats, self.field_stats, self.field_value,
                self.shapes, self.len, self.type)
        return (instanceLoader, (self.__class__, args))

    def _initWithArgs(self, args):
        """Callback for unpickling with :func:`~utilities.utils.instanceLoader`."""

        self.id = args[0]
        self.name = args[1]
        self.parent_type = args[2]
        self.parent_id = args[3]
        self.data_type = args[4]
        self.mask = args[5]
        self.value = args[6]
        self.stats = args[7]
        self.field_stats = args[8]
        self.field_value = args[9]
        self.shapes = args[10]

        # old pickled attributes didn't store the type correctly
        try:
            self.type = args[12]
        except:
            self.type = (self.parent_type, self.name)

        # old pickled attributes didn't store the length, but we need to not
        # bomb on old pickles
        if len(args) > 11:
            self.len = args[11]
        elif self.value is not None:
            try:
                self.len = len(self.value)
            except:
                self.len = 0
        else:
            try:
                # we can get the length from any field func and stat, so just pick one
                self.len = len(self.field_stats[FieldFuncs.Gradient][FieldAttributeStats.Mean])
            except:
                raise

    def _setGraph(self, graph):
        """Reattaced the attribute to its parent object/relation"""

        objs = graph.objects if self.parent_type == 'o' else graph.relations
        self.parent = objs[self.parent_id]
        self.parent.attributes[self.name] = self

        # the type sometimes wasn't pickled correctly, so we have to catch
        # this possible error
        if len(self.type) == 0:
            self.type = (self.parent.type, self.name)

    def _calcStats(self, do_fields=True):
        """Calculates all the statistics and shapes detection on the value of 
        the attribute.

        :param do_fields: if true, calculate the field functions and do shapes
            detection on fielded attributes
        """

        # if its a field we need to run the interpolation on each time step

        #utils.interp(self.value)

        value = self.value
        mask = self.mask
        try:
            self.len = len(value)
        except:
            self.len = 0

        # do shape detection on field attributes
        """
        if 'Field' in self.data_type and do_fields:
            print('finding shapes: ', self.parent.graph.id, repr(self.parent), repr(self))
            self.shapes = findShapes(mask, None)

            volume = map(getVolume, self.shapes)
            STAttribute((self.id[0], 'volume'), '%s_volume' % self.name, self.parent, self.graph,
                        DataTypes.FloatArray, np.array(volume), None)
        """

        def mask_map(func, value=self.value, mask=self.mask, toarray=True):
            """ maps a function over an array using a mask

            :param func: the function to map over each time step of the value
            :param value: the list of timesteps to map over
            :param mask: the list of masks. one for each timestep
            :param toarray: if true convert the result after mapping to a
                numpy array, otherwise leave it as whatever the function 
                returned
            """
            result = []
            for step, mask in zip(value, mask):
                values = step[mask]
                if len(values) > 0:
                    new_value = func(values)
                else:
                    new_value = 0
                result.append(new_value)
            return np.array(result) if toarray else result

        if self.data_type in (DataTypes.FloatArray, DataTypes.IntArray):
            # when ever an attribute of type ndarray is set then
            # we precalculate some stats

            self.stats = {
                AttributeStats.Mean: value.mean(),
                AttributeStats.Std: value.std(),
                AttributeStats.Mode: utils.mode(value),
                AttributeStats.Median: np.median(value),
                AttributeStats.Min: value.min(),
                AttributeStats.Max: value.max(),
            }

        elif self.data_type in (DataTypes.FloatField, DataTypes.IntField):
            # calculate all the basic stats on the scalar fields
            mean = mask_map(np.mean)
            std = mask_map(np.std)
            mode = mask_map(lambda v: utils.mode(v))
            median = mask_map(np.median)
            min = mask_map(np.min)
            max = mask_map(np.max)

            if not do_fields:
                self.field_value = {
                    AttributeStats.Mean: mean,
                    AttributeStats.Std: std,
                    AttributeStats.Mode: mode,
                    AttributeStats.Median: median,
                    AttributeStats.Min: min,
                    AttributeStats.Max: max,
                }
            else:
                self.stats = {
                    AttributeStats.Mean: np.mean(mean),
                    AttributeStats.Std: np.std(std),
                    AttributeStats.Mode: utils.mode(mode),
                    AttributeStats.Median: np.median(median),
                    AttributeStats.Min: np.min(min),
                    AttributeStats.Max: np.max(max),
                }
                self.field_value = {}
                self.field_stats = {}
                _calcGradient(self, mask, value)

        elif self.data_type in (DataTypes.FloatVectorField, DataTypes.IntVectorField):
            mean = mask_map(np.mean)
            std = mask_map(np.std)
            median = mask_map(np.median)
            min = mask_map(np.min)
            max = mask_map(np.max)

            if not do_fields:
                self.field_value = {
                    AttributeStats.Mean: mean,
                    AttributeStats.Std: std,
                    AttributeStats.Mode: None,
                    AttributeStats.Median: median,
                    AttributeStats.Min: min,
                    AttributeStats.Max: max,
                }

            else:
                self.stats = {
                    AttributeStats.Mean: np.mean(mean),
                    AttributeStats.Std: np.std(std),
                    AttributeStats.Mode: None,
                    AttributeStats.Median: np.median(median),
                    AttributeStats.Min: np.min(min),
                    AttributeStats.Max: np.max(max),
                }

                self.field_value = {}
                self.field_stats = {}
                _calcCurl(self, mask, value)
                _calcDivergence(self, mask, value)

        if 'Field' in self.data_type and not getattr(self, 'keep_field_data', False):
            # dump the field data, it takes up to much memory
            self.value = None
            self.mask = None

    def __str__(self):
        return 'STAttribute[id=%s, name=%s, parent=%r, value=(%s:len=%s)%s]' % (
        self.id, self.name, self.parent, self.data_type, self.len, str(self.value)[:40].replace('\n', ''))

    def __repr__(self):
        # if isinstance(self.value, np.ndarray):
        #    size = ', size=%s' % self.value.shape
        # print self.name
        size = ', len=%s' % self.len
        # if 'Vector' in self.data_type:
        #    size = '%s, ndims=%s' % (size, self.dims)
        return 'STAttr[id=%s, name=%s, val=%s%s]' % (self.id, self.name, self.data_type, size)


def _calcFieldStats(masks, field):
    """Calculates field statistics on a scalar field returned by a field function.

    :param masks: list of masks, one for each timestep
    :param field: list of fields, one for each timestep

    :returns: (vals, stats) 
    """

    def mask_map(func, fields):
        """maps a function over a list of fields that have masks

        :param func: the function to map
        :param fields: a list of fields
        """

        result = []
        for field, mask in zip(fields, masks):
            values = field[mask]
            if len(values) > 0:
                new_value = func(values)
            else:
                new_value = 0
            result.append(new_value)
        return np.array(result)

    # vals and stats are the same since we are running on a scalar field
    vals = {
        FieldAttributeStats.Mean: mask_map(np.mean, field),
        FieldAttributeStats.Std: mask_map(np.std, field),
        FieldAttributeStats.Min: mask_map(np.max, field),
        FieldAttributeStats.Max: mask_map(np.min, field),
    }

    stats = {
        FieldAttributeStats.Mean: mask_map(np.mean, field),
        FieldAttributeStats.Std: mask_map(np.std, field),
        FieldAttributeStats.Min: mask_map(np.max, field),
        FieldAttributeStats.Max: mask_map(np.min, field),
    }

    return vals, stats


def _calcVectorFieldStats(masks, vec_field):
    """Calculates field statistics on a vector field returned by a field function.

    :param masks: list of masks, one for each timestep
    :param vec_field: list of fields, one for each timestep

    :returns: (vals, stats) 
    """

    def mag_squared(v):
        """Calculates magnitude squared of 2d and 3d vector field.

        :param v: the vector field to calculate the magnitude off

        :returns: a scalar field of magnitudes
        """
        mags = np.sum(np.square(v), axis=len(v.shape) - 1)
        # print 'mag_squared(%s)=%s' %(v.shape, mags.shape)
        return mags

    def argstat(func, timesteps):
        """Returns a fancy-index that works with numpy for each timestep of the
        vector field which selects either maximum or minimum vector (by magnitude)

        :param func: either np.argmax or np.argmin
        :param timesteps: a list of timesteps (vector fields)
        """

        # magnitude squared of vector fields
        mags = map(mag_squared, timesteps)

        # flat-indices returned arg(min|max)
        indices = []
        for mag, mask in zip(mags, masks):
            # masked array expects False for keeping value, and True for ignore value
            # where as our specification has True for keep and False for ignore,
            # so we invert the mask by doing "-mask"
            masked = maskedarray.masked_array(mag, mask=(-mask))

            # apply the function to the masked array and take the index, which is
            # a single integer, and turn it into a usable index
            indices.append(np.unravel_index(func(masked), mag.shape))

        return indices

    # a list of the maximum vector (by magnitude) of each timestep
    max = [timestep[idx] for timestep, idx in zip(vec_field, argstat(np.argmax, vec_field))]

    # a list of the minimum vector (by magnitude) of each timestep
    min = [timestep[idx] for timestep, idx in zip(vec_field, argstat(np.argmin, vec_field))]

    def mask_map(func, fields):
        """map a function over a list of fields using a mask

        :param func: function to map
        :param fields: list of fields to map over
        """
        return [func(field[mask], axis=0) for field, mask in zip(fields, masks)]

    mean = mask_map(np.mean, vec_field)
    std = mask_map(np.std, vec_field)

    # max, min, mean, std are all scalar vectors
    #    (ie a list of vectors representing one vector per timestep)

    vals = {
        FieldAttributeStats.Mean: np.sum(np.square(mean), axis=1),
        FieldAttributeStats.Std: np.sum(np.square(std), axis=1),
        FieldAttributeStats.Min: np.sum(np.square(min), axis=1),
        FieldAttributeStats.Max: np.sum(np.square(max), axis=1)
    }

    stats = {
        FieldAttributeStats.Mean: np.array(mean),
        FieldAttributeStats.Std: np.array(std),
        FieldAttributeStats.Min: np.array(max),
        FieldAttributeStats.Max: np.array(min),

        AttributeStats.Mean: np.sum(np.square(mean), axis=1).mean(),
        AttributeStats.Std: np.sum(np.square(std), axis=1).std(),
        AttributeStats.Min: np.sum(np.square(min), axis=1).min(),
        AttributeStats.Max: np.sum(np.square(max), axis=1).max(),
    }

    return vals, stats


def component_matrices_to_vector_field(func, timesteps):
    """Some field functions return a matrix for each component/dimension, e.g.
    cx, cy = curl(xy2d_field) and it needs to be turned into a 2d matrix.

    :param func: function to apply to each timestep
    :param timesteps: list of timesteps/fields to apply the function to

    :returns: matrix of combined components
    """

    def field_func(field):
        # apply a field function which returns component matrices
        comp_mats = np.array(func(field))

        # convert component matrices into vector fields
        vf = np.rollaxis(comp_mats, 0, len(comp_mats.shape))

        return vf

    # map field function over each timestep
    timesteps = map(field_func, timesteps)

    return timesteps


def _calcGradient(obj, masks, timesteps):
    """Calculates the gradient and the stat functions of the gradient for fields
    and stores the results in obj.field_value[FieldFuncs.Gradient] and
    obj.field_stats[FieldFuncs.Gradient]

    :param obj: the STAttribute object the field belongs to
    :param masks: a list of masks, one for each timestep
    :param timesteps: the timesteps to calculate the gradient for
    """

    # the gradient is a matrix of values for each dimension, so if
    # the field is 3d, then the gradient returns 3 matrices of the same
    # size as the field. And since its time steps, we get a set for each timestep.
    # For example for a 3d field of size 2x3x4:
    #     gradient = [g1, g2, g3, ..., gn] # the gradient of each timestep
    #     g1 = [x, y, z]  # x is a 2x3x4 matrix of the x-components of the gradient
    #                     # y is a 2x3x4 matrix of the y-components of the gradient, etc.
    # We will turn it into a vector field of size 2x3x4x3

    grad = component_matrices_to_vector_field(gradient, timesteps)

    vals, stats = _calcVectorFieldStats(masks, grad)
    obj.field_value[FieldFuncs.Gradient] = vals
    obj.field_stats[FieldFuncs.Gradient] = stats


def _calcDivergence(obj, masks, timesteps):
    """Calculates the divergence and the stat functions of the gradient for fields
    and stores the results in obj.field_value[FieldFuncs.Divergence] and
    obj.field_stats[FieldFuncs.Divergence]

    :param obj: the STAttribute object the field belongs to
    :param masks: a list of masks, one for each timestep
    :param timesteps: the timesteps to calculate the divergence for
    """

    div = map(divergence, timesteps)

    vals, stats = _calcFieldStats(masks, div)
    obj.field_value[FieldFuncs.Divergence] = vals
    obj.field_stats[FieldFuncs.Divergence] = stats


def _calcCurl(obj, masks, timesteps):
    """Calculates the curl and the stat functions of the gradient for fields
    and stores the results in obj.field_value[FieldFuncs.Curl] and
    obj.field_stats[FieldFuncs.Curl]

    :param obj: the STAttribute object the field belongs to
    :param masks: a list of masks, one for each timestep
    :param timesteps: the timesteps to calculate the curl for
    """

    c = component_matrices_to_vector_field(curl, timesteps)
    vals, stats = _calcVectorFieldStats(masks, c)
    obj.field_value[FieldFuncs.Curl] = vals
    obj.field_stats[FieldFuncs.Curl] = stats