"""
This is the base class for all distinctions. To create new distinction you
must subclass :class:`.Distinction` and implement the following methods:

    * :meth:`.Distinction.splitGraphs`
    * :meth:`.Distinction.getRandomDistinction`
    * :meth:`.Distinction.getNumberOfSubtypes`
    * :meth:`.Distinction.getType`
    * :meth:`.Distinction.getReferencedTypes`
    * :meth:`.Distinction.getSplitType`

If you are creating a conjugate distinction then these methods must also be
implemented by your subclass:    

    * :meth:`.Distinction.getNumberOfBaseDistinctionsNeeded`
    * :meth:`.Distinction.getBaseDistinctions`

If your distinction is not-valid for all possible schemas, then 
:meth:`.Distinction.isValidForSchema` must also be implemented. For example,
:class:`~srpt.distinctions.FieldAttributeDistinction` is only valid for schemas
that have fielded attributes.    

"""

from utilities.Enum import Enum
from utilities.utils import instanceReducer


class InvalidDistinctionException(Exception):
    """Thrown when something goes very wrong.
    """

    pass


class AbstractMethodException(Exception):
    """Exception thrown when a subclass fails to implement one of the required 
    methods."""

    def __init__(self, cls, method=None, *args, **kw):
        if method is None:
            msg = "Method is Abstract For: %s" % (cls)
        else:
            msg = "%s.%s is Abstract!" % (cls, method)
        Exception.__init__(self, msg, *args)
        #Exception.__init__(self, msg, *args, **kw)


#: An enum of the different types of distinctions, so that distinctions can declare their type
#: and the distinction generator can intelligently select which distinctions are valid
#:
#:    * Basic - a basic type of distinction, its not temporal or spatial or conjugate,
#:      just a plain distinction like :class:`Exists <srpt.distinctions.ExistsDistinction.ExistsDistinction>`
#:      or :class:`Attribute <srpt.distinctions.AttributeDistinction.AttributeDistinction>`
#:    * Temporal - a distinction that focuses on the temporal aspect of data,
#:      such as :class:`Temporal Exists <srpt.distinctions.TemporalExistsDistinction.TemporalExistsDistinction>`
#:    * Spatial - a distinction that focuses on the spatial aspect of data,
#:      such as :class:`Field <srpt.distinctions.FieldDistinction.FieldDistinction>`
#:    * Conjugate - combines multiple non-conjugate distinctions together, such as
#:      :class:`Count Conjugate <srpt.distinctions.CountConjugateDistinction.CountConjugateDistinction>`
#:      or :class:`Boolean <srpt.distinctions.BooleanDistinction.BooleanDistinction>`
#:
#: Types can be combined such as :class:`Temporal Ordering <srpt.distinctions.TemporalOrderingDistinction.TemporalOrderingDistinction>`
#: which is (Temporal, Conjugate).
DistinctionTypes = Enum('DistinctionTypes', 'Basic', 'Conjugate', 'Temporal', 'Spatial')


class Distinction(object):
    """The base class for all distinctions."""

    def __reduce__(self):
        """Default reducer for distinctions. Expects all distinctions to follow
        the :class:`~utilities.utils.instanceReducer` protocol.
        """

        return instanceReducer(self)

    def splitGraphs(self, graphs):
        """Split the graphs into two sets, those matching the distinction and
        those not matching. Which graphs are which are stored in a
        :class:`~srpt.distinctions.Split.Split` object which is then returned.

        :param graphs: the graphs to split
        :type graphs: Dict[graph.id => :class:`~srpt.STGraph.STGraph`]

        :returns: a split of the graphs into a yes and no set
        :rtype: :class:`srpt.distincitons.Split.Split`

        .. note::

            This is an abstract method and must be overridden
        """

        raise AbstractMethodException(self.__class__)

    def getBaseDistinctions(self):
        """For conjugate distinctions this should be overridden and return the
        base distinctions used. For none conjugate it will automatically return
        an empty list.

        :returns: list of base distinctions used by this distinction

        """

        return []

    def getReferencedTypes(self):
        """Returns the objects, relations and/or attributes type used by this 
        distinction, it must be over-ridden.

        .. note::

            This is an abstract method and must be overridden
        """

        raise AbstractMethodException(self.__class__)

    def getSplitType(self):
        """Returns a tuple of information about the split, such as the stat
        function for AttributeDistinction. The first should be a plain English 
        name of the distinction, other elements are distinction dependent.

        .. note::

            This is an abstract method and must be overridden
        """

        raise AbstractMethodException(self.__class__)

    @staticmethod
    def getRandomDistinction(config, graphs, *base_distinctions):
        """Generates a random distinction of this type than is valid for the
        schema *config.schema* and for the given *graphs*.

        This function for *must* take *graphs* as its first argument, and if its
        a conjugate distinction it *must* then take, as separate args, not a tuple,
        the base distinctions it should use. For example:: 

            BooleanDistinction.getRandomDistinction(a_config, some_graphs, first_base_distinction, second_base_distinction)

        :param config: a :class:`~utilities.Config.Config` object
        :param graphs: the graphs to use for generating a random distinction
        :type graphs: Dict[graph.id => :class:`~srpt.STGraph.STGraph`]
        :param \*base_distinctions: the base distinctions to be used

        :returns: a random distinction

        .. note::

            This is an abstract method and must be overridden
        """
        raise AbstractMethodException(Distinction)

    @staticmethod
    def getNumberOfSubtypes(config, low_estimate=True):
        """Get an estimate of the number of different sub-types for this distinction.
        This is used to estimate a PDF for randomly sampling the distinction space.
        Examine the code of other distinctions to get a feel for how things are
        estimated. 

        :param config: a :class:`~utilities.Config.Config` object
        :param low_estimate: if ``True`` then a very rough estimate is used,
            otherwise a more accurate estimation is returned

        :returns: estimate of the number of sub-types

        .. note::

            This is an abstract method and must be overridden
        """

        raise AbstractMethodException(Distinction)

    @staticmethod
    def getType():
        """Return the type of distinction as a tuple of  :data:`DistinctionTypes`. 
        For instance a temporal conjugate distinction would return::

            (DistinctionTypes.Temporal, DistinctionTypes.Conjugate)

        .. note::

            This is an abstract method and must be overridden
        """

        raise AbstractMethodException(Distinction)

    @staticmethod
    def getNumberOfBaseDistinctionsNeeded():
        """For a conjugate distinction return the number of base distinctions it
        needs to operate, and expects in the constructor.

        Note: Only conjugate distinctions need to implement this. I.e. those that
        return :data:`DistinctionTypes.Conjugate <.DistinctionTypes>` when 
        :meth:`.getType` is called.

        :returns: the number of base distinctions needed to create this distinction    
        :rtype: Integer

        .. note::

            This is an abstract method and must be overridden
        """

        raise AbstractMethodException(Distinction)

    @staticmethod
    def isValidForSchema(schema):
        """Given a *schema* return True if this type of distinction is valid
        for the schema. Default is True. Should be overridden if there are any
        schemas a distinction is not valid for.

        :returns: True if this distinction type is valid for the *schema*, False
            otherwise
        """

        return True