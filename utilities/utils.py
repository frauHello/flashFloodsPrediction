"""
This module contains a variety of convenience and utility functions used
throughout the SRPT framework.
"""

import sys

from collections import defaultdict
from itertools import chain
import pickletools
import time
import os
import stat
from numpy import nanmean
import numpy as np
import functools
import pandas as pd
import inspect

try:
    import scipy.stats
except ImportError:
    print('Missing scipy.stats, functionality will be limited')


def fsize(fname):
    """Return the size in bytes of the given file."""
    return os.stat(fname)[stat.ST_SIZE]


def argparser(arg_func):
    """An annotation that indicates which function will create and return an argparse
    parser for use of a function:

    .. code-block:: python

        def _make_parser():
            # make an argparse parser
            parser = ...
            return parser

        @argparser(_make_parser)
        def main(parser):
            ...

    This is done so that the documentation of the CLI can be handled automatically.
    The parser created will be automatically passed in as the *first* argument 
    of the function in addition to any other arguments passed in at call time. 
    For a full example of how and more detail on why see the "Writing New
    Scripts" section of :doc:`graphing_and_scripts`.
    """

    # get the parser
    parser = arg_func()
    parser_doc = '\n\n::\n\n\t' + '\n\t'.join(parser.format_help().split('\n'))

    def doc_modifier(func):
        fdoc = func.__doc__
        func.__doc__ = ('' if fdoc is None else fdoc) + parser_doc

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(parser, *args, **kwargs)

        return wrapper

    return doc_modifier


def interp(m, copy=False):
    """Interpolates over 1,2,3-d arrays using mean of non-NaN neighbors to replace NaNs.

    Note: This function needs to be reworked to provide more accurate interpolation
    instead of mere mean of nearest neighbors.
    """
    ###m=self.value
    if copy:
        m = m.copy()

        # early break if no nans found
    nans = np.where(np.isnan(m) == True)
    if len(nans) == 0 or len(nans[0]) == 0:
        return m

    # build up the neighborhood indexing array
    deltas = [slice(-1, 2, 1) for i in m.shape]
    deltas = np.mgrid[deltas]

    for cords in zip(*nans):
        # the neighborhood selection grid
        grid = [cords[i] + deltas[i] for i in range(len(cords))]

        # wrap the edges
        for i, g in enumerate(grid):
            g[g == m.shape[i]] -= m.shape[i]

        # average the neighborhood
        m[cords] = nanmean(m[grid], axis=None)

    return m


def product(*args, **kwargs):
    """Iterates over the Cartesian product of the input iterables.

       # >>> for x, y in product(range(3), range(2)):
        #>>>     print x, y
        0, 0
        0, 1
        1, 0
        1, 1
        2, 0
        2, 1
    """

    # this is a recursive implementation that uses minimal memory compared to the
    # implementation commented outbelow which pre-computes all products.
    # Note: Python 2.6 has a C implementation of this function

    head = kwargs.get('head', tuple())
    depth = kwargs.get('depth', 0)

    for item in args[depth]:
        if depth == len(args) - 1:
            ret = tuple(chain(head, (item,)))
            yield ret
        else:
            next_head = tuple(chain(head, (item,)))
            for i in product(*args, **{'head': next_head, 'depth': depth + 1}):
                yield i


# def product(*args, **kwds):
#    """Cartesian product of input iterables.
#    From: http://docs.python.org/library/itertools.html#itertools.product
#    """
#
#    pools = map(tuple, args) * kwds.get('repeat', 1)
#    result = [[]]
#    for pool in pools:
#        result = [x+[y] for x in result for y in pool]
#    for prod in result:
#        yield tuple(prod)

def mode(v):
    """Calculates the mode of an array, returns ``None`` if array is empty, or 
    if an error occurs.

    Note: this uses :func:`scipy.stats.mode`, but only returns a single value 
    instead of a list of modes.
    """

    if len(v) > 0:
        return scipy.stats.mode(v)[0][0]
    else:
        return None


def flipCoin():
    """Returns True or False uniformly distributed . . .  like flipping a coin."""

    return np.random.randint(2) > 0


def cleandict(adict):
    """Cleans the keys and values of the dict and returns a cleaned dict 
    to ensure there are numpy types in it. This is most often used to ensure 
    the pickled files don't have numpy dependency which can be annoying due to
    numpy version differences.

    :returns: a dict with clean keys and values
    """

    newdict = {}
    for k, v in adict.items():
        newdict[np2py(k)] = np2py(v)
    return newdict


def np2py(val):
    """Convert (if needed) a numpy type to a python type."""

    if isinstance(val, np.ndarray):
        return tuple(val)
    elif hasattr(val, 'dtype'):
        if val.dtype.kind == 'i':
            val = int(val)
        elif val.dtype.kind == 'f':
            val = float(val)
        elif val.dtype.kind == 'S':
            val = str(val)
    return val


class _OldStyle: pass


def instanceLoader(clazz, init_args):
    """Used by reduce as the "unreducer" returned to pickle when pickling
    objects. It is generally used by :func:`.instanceReducer` instead of being
    called manually. 

    If the class defines *_initWithArgs*, it is called using the
    variables stored in *_init_args* as its parameters. Note, when *_initWithArgs*
    is called *__init__* is never called. The object is created *__new__* instead.
    Otherwise, if no *_initWithArgs* is defeined, the class's *__init__* function
    is called instead (also using *_init_args* as its parameters).

    Example that calls *__init__* on unpickling

    .. code-block:: python

        class Foo(object):
            _init_args = ('foo', 'bar')

            def __init__(self, foo, bar):
                ...

            def __reduce__(self):
                # on unpickling this will call:
                #    Foo(pickled_foo, pickled_bar)
                return instanceReducer(self)

    Or an *_initWithArgs* function could have been added which would have been
    called instead of *__init__*:

    .. code-block:: python

        class Foo(object):
            _init_args = ('foo', 'bar')

            def _initWithArgs(self, args):
                ...

            def __reduce__(self):
                # on unpickling this will call:
                #    empty_foo._initWithArgs(pickeled_init_args)
                return instanceReducer(self)

    Using *_init_args* and *_initWithArgs* allows for much greater control over
    what gets pickled and how it gets unpickled than letting Python handle it
    all on its own. 
    """

    if hasattr(clazz, '_initWithArgs'):
        if hasattr(clazz, '__new__'):
            obj = clazz.__new__(clazz)
        else:
            obj = _OldStyle()
            obj.__class__ = clazz
        obj._initWithArgs(init_args)
    else:
        obj = clazz(*init_args)

    return obj


def instanceReducer(obj):
    """Reduces an instance that defines *_init_args* to a tuple returnable by
    *__reduce__*. This is more convenient than forming the tuple for *__reduce__*
    manually. 

    .. warning::
        While using *instanceReducer* makes the pickles smaller, it means that
        changing '_init_args' can break unpickling of previously pickled objects.
        Since no dict is being saved the length and order of the args is fixed
        as its just a tuple.

    .. code-block:: python

        class Foo(object):
            _init_args = ('foo', 'bar')

            def __reduce__(self):
                # only self.foo and self.bar will be pickled, regardless of what
                # else might be in the dictionary
                return instanceReducer(self)
    """

    args = []
    for k in obj.__class__._init_args:
        v = getattr(obj, k)
        #print(k)
        #print(type(v))
        args.append(v)
    return (instanceLoader, (obj.__class__, tuple(args)))


def loadPickledVars(pickle_file, vars_list):
    """Given a pickle filename and a list variable names, scan through the pickle
    file and extract only those variables, anywhere they are found in the pickled
    object(s). A dict is returned of {variable_name -> value}. This is much faster
    than loading the pickle as no objects are constructed during the process.
    However, the only variables that can be loaded in this way are int, float,
    and str.

    .. warning::

        This method is very experimental!
    """

    vars = dict.fromkeys(vars_list, [])
    cur_var = None
    try:
        for opcode, arg, pos in pickletools.genops(open(pickle_file, 'rb')):
            if cur_var is None and opcode.name in ('SHORT_BINSTRING', 'STRING') and arg in vars:
                cur_var = arg
            elif cur_var is not None and opcode.name in (
            'BINFLOAT', 'BININT', 'SHORT_BINSTRING', 'FLOAT', 'INT', 'STRING'):
                vars[cur_var].append(arg)
                cur_var = None
    except ValueError:
        print(pickle_file)
        print('size:', formatBytes(os.stat(pickle_file)[stat.ST_SIZE]))
        raise

    return vars


def randomIter(seq):
    """Randomly iterator over an iterable, attempts to be as memory efficient as
    possible.
    """

    if len(seq) == 0:
        raise IndexError('Sequence is 0 length')

    idxs = np.arange(len(seq))
    np.random.shuffle(idxs)
    #print("the type of the gotten sequence is")
    #print(type(seq))
    if type(seq) is list:
        return (seq[idx] for idx in idxs)
    elif isinstance(seq, (dict, defaultdict)):
        keys = list(seq.keys())
        return (seq[keys[idx]] for idx in idxs)
    elif type(seq) is set:
        # raise ValueError('Sets are not randomly iterable . . . yet!')
        return (list(seq)[idx] for idx in idxs)
    else:
        raise ValueError('randomIter not implemented for type: %s', type(seq))


class TimeIt(object):
    """Simply timing object. Called with a label to start and stop the timing for
    a *label*. The same label can be called multiple times to accumulate time.

       # >>> timer = TimeIt()
        #>>> timer('foo')
        #>>> ... do work
        #>>> timer('foo')
        #>>> timer('bar')
        #>>> ... do work
        #>>> timer('bar')
        #>>> print timer
        TimeIt[
            foo = 12s
            bar = 1m32s
        ]
       # >>> timer.bar
        92    
    """

    def __init__(self):
        self.__order = []

    def __call__(self, name):
        t = time.time()
        t_name = '__%s' % name
        if not name in self.__dict__:
            self.__order.append(name)

        if self.__dict__.get(t_name, None) is None:
            self.__dict__[t_name] = t
            self.__dict__.setdefault(name, 0)
        else:
            self.__dict__[name] = self.__dict__.get(name, 0) + (t - self.__dict__[t_name])
            self.__dict__[t_name] = None

    def __str__(self):
        s = 'TimeIt[\n'
        for name in self.__order:
            s = '%s\t%s = %s\n' % (s, name, formatTime(self.__dict__[name]))
        s = '%s]' % s
        return s


def dictFromKeys(full_dict, keys_to_extract, skip_missing=False):
    """Given a dictionary, *full_dict*, return a smaller dictionary containing
    only the key/value pairs that are contained in *keys_to_extract*. If
    *skip_missing* is ``True`` keys in *keys_to_extract*, but not in *full_dict*
    are simply skipped instead of raising a *KeyError* exception.
    """

    if skip_missing:
        return dict(((key, full_dict[key]) for key in keys_to_extract if key in full_dict))
    else:
        return dict(((key, full_dict[key]) for key in keys_to_extract))


def maxKey(adict):
    """Returns the key associated with maximum entry in the dict.

    Assumes that all keys are comparable using the greater than *>* operator.
    """

    max_key, max_value = None, -sys.maxsize
    for key, value in adict.items():
        if value > max_value:
            max_value = value
            max_key = key

    return max_key


def splitDict(adict, count, random=False, seed=None):
    """Splits a dictionary into to two smaller dictionaries. 

    :param count: the number of items to be in the first dictionary to return, 
        the remaining items are in the second dictionary
    :param random: should the splitting be randomized, if ``False`` than the split
        is consistent between multiple calls, if *seed* is also given, then use
        that to seed the random generator
    :param seed: If ``None`` then use the current time as the seed, otherwise
        it will be used as the seed when *random* is ``True``.

    :returns: two dictionaries *front, back* such that ``len(front) == count``
    """

    keys = adict.keys()

    if random:
        seed = int(time.time()) if seed is None else seed
    else:
        if seed is None:
            seed = 112358

    rs = np.random.RandomState(seed)
    keys=list(keys)
    rs.shuffle(keys)

    front = dict(((k, adict[k]) for k in keys[:count]))
    back = dict(((k, adict[k]) for k in keys[count:]))

    return front, back


def foldDict(adict, num_folds, random=False, seed=11235813):
    """Fold the dictionary, splitting it into num_folds even sized chunks.
    Dictionary folds are stable, even when randomized, due to using
    a constant seed. This way multiple calls to foldDict() with the same
    dictionary will return the same folds. seed can be set to None to
    use a random seed."""
    fold_size = int(len(adict) / num_folds)

    keys = adict.keys()
    if random:
        seed = time.time() if seed is None else seed
    else:
        if seed is None:
            seed = 112358
    rs = np.random.RandomState(seed)
    rs.shuffle(keys)

    folds = []
    for fold in range(num_folds):
        folds.append(dict(((k, adict[k]) for k in keys[fold * fold_size:(fold + 1) * fold_size])))

    return folds


class SimpleContainer(object):
    """A simple class that combins dict bracket indexing and dot indexing. 
    ``cont.foo`` is the same as ``cont['foo']``.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, value):
        self.__dict__[name] = value

    def __str__(self):
        s = 'SimpleContainer[\n'
        for k, v in self.__dict__.items():
            s = '%s\t%s=%s,\n' % (s, k, v)
        s = '%s]\n' % s
        return s


def randomChoice(seq, key=False):
    """Similar to *random.choice* but handles dicts, sets, and defaultdicts also,
    though they are not constant time access like lists are. If key=True return
    the key from a dictionary instead of the value"""
    #filter_to_list=list(seq)
    if len(seq) == 0:
        raise IndexError('Sequence is 0 length')

    # pick a random index to return
    idx = int(np.random.random() * len(seq))
    if isinstance(seq, (dict, set, defaultdict)):
        # for non-lists we need an iterable
        if type(seq) == set:
            # for sets the set itself is iterable
            items = seq
        else:
            if hasattr(seq, '_itervalues'):
                items = seq._itervalues()
            else:
                # for dicts, and defaultdicts use itervalues() as the iterable
                if key:
                    items = seq.iterkeys()
                else:
                    items = seq.itervalues()

        # iterate over the values until we get to the correct index and then return it
        i = 0
        for item in items:
            if i == idx:
                return item
            i += 1
    else:
        # lists allow direct indexing so just return the item
        return seq[idx]


def formatTime(seconds):
    """Formats the number *seconds* into "h:mm:ss.milis"

      #  >>> formatTime(6412.12)
        '1:46:52.12'
    """

    h = int(seconds / (3600))
    m = int((seconds % 3600) / 60)
    s = (seconds % 60)

    return '%i:%02i:%0.2f' % (h, m, s)


def formatBytes(num):
    """Formats a number of bytes into a human readable format
    (e.g. 1024 -> 1KB)

      #  >>> formatBytes(1234)
        '1.2KB'    
       # >>> formatBytes(123456)
        '120.6KB'
       # >>> formatBytes(12345678)
        '11.8MB'
    """

    for size in ('bytes', 'KB', 'MB', 'GB', 'TB'):
        if num < 1024.0:
            return "%3.1f%s" % (num, size)
        num /= 1024.0


def dynamicImport(name):
    """Dynamically import in a more usable manner than __import__."""
    #print("INSIDE DYNAMIC IMPORT")
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        #print("comp:%s"%comp)
        #print("mod:%s"%mod)
        mod = getattr(mod, comp)
        #print("This is the the stuff we will import")
        #print(mod)

    if(not inspect.isclass(mod)):
        mod = getattr(mod, comp)
        #print("This is the the stuff we will import")
        #print(mod)

    return mod


def weightedChoiceIndex(cdf):
    """Randomly selects from the list using the probabilities in *cdf*
    (assumed to be a CDF) and returns the index.
    """

    rand = np.random.random()
    index = np.searchsorted(cdf, rand)

    return index


def weightedChoice(choices, cdf):
    """Randomly selects from the list using the probabilities in *cdf*
    (assumed to be a cdf)
    """

    if len(cdf) != len(choices):
        raise ValueError('CDF length mismatch with length of choices: %s != %s' % (len(cdf), len(choices)))

    rand = np.random.random()
    index = np.searchsorted(cdf, rand)

    if index >= len(choices):
        raise ValueError('CDF produced index greater than length of choices: %s >= %s rand=%s cdf=%s' % (
        index, len(choices), rand, cdf))

    return choices[index]


def curl(uvw):
    """Computes the curl of u, v, w return curl_x, curl_y, curl_z. 
    Assumes regular grid spacing with a distance of 1. *uvw* is a numpy array.
    """

    if len(uvw.shape) == 3:
        u = uvw[:, :, 0]
        v = uvw[:, :, 1]
        w = None
    else:
        u = uvw[:, :, :, 0]
        v = uvw[:, :, :, 1]
        w = uvw[:, :, :, 2]

    if w is None:
        junk, Fu_y = gradient(u)
        Fv_x, junk = gradient(v)

        # 2d curl is just an amount of rotation about the z-vector
        phi = Fv_x - Fu_y

        # hence we need to actually rotate a field of basis vectors so
        # that we can return a vector field
        x = np.ones_like(u)
        y = np.zeros_like(u)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        curlx = x * cos_phi - y * sin_phi
        curly = x * sin_phi + y * cos_phi

        # cav = .5 * curlx
        # return curlx, cav



        return curlx, curly

    else:
        junk, Fu_y, Fu_z = gradient(u)
        Fv_x, junk, Fv_z = gradient(v)
        Fw_x, Fw_y, junk = gradient(w)

        curlx = Fw_y - Fv_z
        curly = Fu_z - Fw_x
        curlz = Fv_x - Fu_y

        nrm = np.sqrt(u ** 2 + v ** 2 + w ** 2)

        # cav = .5 * (curlx * u + curly * v + curlz * w) / nrm;
        # return curlx, curly, curlz, cav

        return curlx, curly, curlz


def divergence(uvw):
    """Computes the divergence of F=[u, v, w] return div(F). The divergence of a
    vector-field is a scalar field of the divergence at each point. Assumes
    regular gridspacing with a distance of 1. *uvw* is a numpy array.
    """

    # print uvw.shape
    if len(uvw.shape) == 3:
        u = uvw[:, :, 0]
        v = uvw[:, :, 1]
        w = None
    else:
        u = uvw[:, :, :, 0]
        v = uvw[:, :, :, 1]
        w = uvw[:, :, :, 2]

    if w is None:
        Fu_x, junk = gradient(u)
        junk, Fv_y = gradient(v)

        return Fu_x + Fv_y
    else:
        Fu_x, junk, junk = gradient(u)
        junk, Fv_y, junk = gradient(v)
        junk, junk, Fw_z = gradient(w)

        return Fu_x + Fv_y + Fw_z


def gradient(f):
    """Computes the gradient of the field. *f* is a numpy array.
    """

    if len(f.shape) == 2:
        try:
            singular = 1 in f.shape

            # if a dimension happens to be 1, ie in f.shape = (2,1,3), then an error will
            # be thrown, so we have to do some gymnastics.
            if singular:
                # create an index into the array which will remove all size 1 dimensions
                index = [0 if size == 1 else slice(None) for size in f.shape]

                # run the gradient on the reduced array
                raw_g = np.gradient(f[index])

                # we need to fill in the full gradient which will have more parts than the
                # reduced gradient
                g = [None] * len(f.shape)

                # reshape each part of the gradient to the original shape
                g_i = 0
                for i in range(len(f.shape)):
                    if f.shape[i] == 1:
                        # singular dimension is all zeros
                        g[i] = np.zeros(f.shape)
                    else:
                        # non-singular gets the calculated gradient
                        if type(raw_g) == np.ndarray and len(raw_g.shape) == 1:
                            g[i] = raw_g.reshape(f.shape)
                        else:
                            g[i] = raw_g[g_i].reshape(f.shape)

                        g_i += 1

            else:
                # no singular dimension means we can jus run the thing
                g = np.gradient(f)

            gx, gy = g
        except:
            print(f)
            #print('Shape:', f.shape)
            raise
        return gx, gy

    elif len(f.shape) == 3:
        try:
            singular = 1 in f.shape

            # if a dimension happens to be 1, ie in f.shape = (2,1,3), then an error will
            # be thrown, so we have to do some gymnastics.
            if singular:
                # create an index into the array which will remove all size 1 dimensions
                index = [0 if size == 1 else slice(None) for size in f.shape]

                # run the gradient on the reduced array
                raw_g = np.gradient(f[index])

                # we need to fill in the full gradient which will have more parts than the
                # reduced gradient
                g = [None] * len(f.shape)

                # reshape each part of the gradient to the original shape
                g_i = 0
                for i in range(len(f.shape)):
                    if f.shape[i] == 1:
                        # singular dimension is all zeros
                        g[i] = np.zeros(f.shape)
                    else:
                        # non-singular gets the calculated gradient
                        if type(raw_g) == np.ndarray and len(raw_g.shape) == 1:
                            g[i] = raw_g.reshape(f.shape)
                        else:
                            g[i] = raw_g[g_i].reshape(f.shape)

                        g_i += 1

            else:
                # no singular dimension means we can just run the thing
                g = np.gradient(f)

            gx, gy, gz = g
        except:
            print(f)
            print('Shape:', f.shape)
            raise
        return gx, gy, gz


def danger_level(label):
    danger = 50
    if (label <0.1):
        danger = 4000
    elif (0.1 <= label < 0.2):
        danger = 3500

    elif (0.2 <= label < 0.3):
        danger = 3000

    elif (0.3 <= label < 0.4):
        danger = 2500

    elif (0.4 <= label < 0.5):
        danger = 2000

    elif (0.5 <= label < 0.7):
        danger = 1500

    elif (0.7 <= label < 0.9):
        danger = 1000

    elif (0.9 <= label):
        danger = 200

    return danger