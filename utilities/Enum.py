"""
Simple Enum object based on strings instead of ints. Instead of defining
EnumValue objects, the strings pickle very fast and small. Strings also unpickle
uniquely to a given enum which EnumValues can do, but ints can't.

The standard colors example:
    #>>> mycolors = Enum('Colors', 'Red', 'Green', 'Blue')
    #>>> print type(mycolors.Red), mycolors.Red
    <type 'str'> Colors.Red

Supports indexing, __contains__, and case-insensitive name-to-enumvalue:
   # >>> mycolors['Red'] # case-sensitive
    Colors.Red
    #>>> favcolor = mycolors.Green
    #>>> favcolor in mycolors
    True
    #>>> "Colors.Red" in mycolors
    True
    #>>> mycolors.get('rEd') #case-INsensitive
    Colors.Red


"""

from utilities.utils import randomChoice
import sys

class Enum(object):
    def  __init__(self, enum_name, *names):
        # helper functions to get around the name-munging done by python
        # and interfered with by custom __get/setattr__

        # initially enums are mutable so we can set them up
        self.__dict__['_Enum__immutable'] = False

        self.__name = enum_name
        self.__names = set()

        # populate the enum
        for name in names:
            fullname = sys.intern('%s.%s' % (enum_name, name))
            setattr(self, sys.intern(name), fullname)
            self.__names.add(fullname)
        self.__doc__ = 'Enum[%s][%s]' % (enum_name, names)

        # make the enum immutable
        self.__immutable = True

    def pickRandom(self, exclude=[]):
        """Chooses a random enumeration for this Enum object excluding
        the given values as options::
           # >>> # picks a random color other than Red
           # >>> mycolors.pickRandom([mycolors.Red]
            'Colors.Blue'

        :param exclude: a list of enum values to exclude

        :returns: a randomly selected enum value
        """

        names = [name for name in self.__names if name not in exclude]
        item = randomChoice(names)
        return item

    def rawName(self, enum_value):
        """Returns the name of the enum value with out the enum's
        name prepended::
           #  >>> myColors.rawName('Colors.Red')
             'Red'

        :param enum_value: the enum_value to get the simple/raw name of

        :returns: the simple/raw name
        """

        return enum_value[len(self.__name) + 1:]

    def get(self, name):
        """Returns the enum value having the given name, this does a case-insensitive
        search, so is slightly slower than index notation::
          #  >>> mycolors.get('rEd')
            Colors.Red

        :param name: the name of the enum value to return

        :returns: the enum value named *name*

        :throws: KeyError if no matching enum is found
        """

        idx = len(self.__name) + 1
        for name_ in self.__names:
            if name_[idx:].lower() == name.lower():
                return name_

        raise KeyError('%s not found in Enum %s' % (name, self.__name))

    def __getitem__(self, name):
        """Allows for dict-like access::
            mycolors['Red'] -> Colors.Red
        """

        try:
            return getattr(self, name)
        except:
            raise

        return None

    def __contains__(self, name):
        """Allows using the ``in`` keyword with enums::
            favcolor = mycolors.Red
            favcolor in mycolors -> True
        """

        return name in self.__names

    def __iter__(self, name):
        """Returns an iterator of the names of the enum values::
            list(mycolors) -> ['Red', 'Green', 'Blue']
        :returns: an iterator over the names
        """

        return iter(self.__names)

    def __setattr__(self, name, value):
        # check if the enum is immutable yet. during init its annoying to not
        # be able to set things directly, so immutability is a flag
        if self.__immutable:
            raise Exception('Enums are immutable')
        super(Enum, self).__setattr__(name, value)

    def __len__(self):
        return len(self.__names)

    def __repr__(self):
        return '%s[%s]' % (self.__name, self.__names)
