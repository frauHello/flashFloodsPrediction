import copy


from collections import MutableMapping

class OrderedDict(dict, MutableMapping):
    def __init__(self, *args, **kwds):
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))

        try:
            _ = self.__end
        except AttributeError:
            self.clear()

        self.update(*args, **kwds)

    def clear(self):
        end = []
        self.__end = end
        end += [None, end, end]  # sentinel node for doubly linked list
        self.__map = {}  # key --> [key, prev, next]
        dict.clear(self)

    def __setitem__(self, key, value):
        if key not in self:
            end = self.__end
            curr = end[1]
            curr[2] = end[1] = self.__map[key] = [key, curr, end]
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        # print '__delitem__(%s)' % (key)
        # print '\tself:', self.keys()
        dict.__delitem__(self, key)
        key, prev, next = self.__map.pop(key)
        prev[2] = next
        next[1] = prev

    def __iter__(self):
        end = self.__end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.__end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def popitem(self, last=True):
        if not self:
            raise KeyError('dictionary is empty')
        key = next(reversed(self)) if last else next(iter(self))
        value = self.pop(key)
        return key, value

    def __reduce__(self):
        items = [[k, self[k]] for k in self]
        tmp = self.__map, self.__end
        del self.__map, self.__end
        inst_dict = vars(self).copy()
        self.__map, self.__end = tmp
        if inst_dict:
            return (self.__class__, (items,), inst_dict)
        return self.__class__, (items,)

    def keys(self):
        return list(self)

    def deepcopy(self, nocopy=[]):
        """"Performs a deepcopy of the OrderedDict object. The nocopy parameter is list/tuple of
        keys to not copy over."""

        return self.__deepcopy__(nocopy=nocopy)

    def __deepcopy__(self, memo=None, _nil=[], nocopy=[]):
        #
        # We implement our own deep copy due to some strange behavior when using the default
        # deepcopy from the copy module. In addition we add the ability to specify keys to
        # not copy in the process.
        #

        if memo is None:
            memo = {}

        newdict = self.__class__()
        memo[id(self)] = newdict

        for key, item in self.__dict__.items():
            if key in nocopy: continue

            if '_OrderedDict__' not in key:
                setattr(newdict, key, copy.deepcopy(item, memo))

        for key, item in self.iteritems():
            if key in nocopy: continue

            newdict[copy.deepcopy(key, memo)] = copy.deepcopy(item, memo)

        return newdict

    setdefault = MutableMapping.setdefault
    update = MutableMapping.update
    pop = MutableMapping.pop
    values = MutableMapping.values
    items = MutableMapping.items
    iterkeys = MutableMapping.keys
    """ 
    itervalues = MutableMapping.itervalues
    iteritems = MutableMapping.iteritems
     """
    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, self.items())

    def copy(self):
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    def __eq__(self, other):
        if isinstance(other, OrderedDict):
            return len(self) == len(other) and \
                   all(p == q for p, q in zip(self.items(), other.items()))
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other