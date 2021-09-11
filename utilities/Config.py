"""Config objects are used to keep data about the current experiment and 
learning/training parameters in one place. Generally, a single one is created
when the program starts (usually by loading an experiment file and augmenting
it with command line options), and it is then passed around so that everyone
has access to the same parameters. This is done instead of using GLOBAL variables
which aren't global between modules anyways. 

A good example of the use of a config object is found in
:func:`experiments.ExperimentLoader.loadExperimentFile`.
"""

import os

import argparse
from utilities.utils import formatBytes
from utilities.OrderedDict import OrderedDict


class ConfigLoadAction(argparse.Action):
    """Used to load files specified on the command line into a config object."""

    def __call__(self, parser, namespace, values, option_string=None):
        Config.fromFile(filename=values, config=namespace)


class Config(OrderedDict):
    """A simple class that acts like an object and dict as well as being able to
    load its values from a simple key=value pairs file. Note that the value's are
    insecure as they get evaluated as python expressions with the locals being
    the previously parsed values. This allows such constructs as::

        foo = 42
        bar = "this is a string"
        alist = [foo, bar] # python comments are allowed in-line
        foobar = "fun with python = %s" % (alist)

    Evalating the values as python statements makes loading a config from a
    file very powerful.
    """

    def __init__(self, *args, **kwargs):
        """Creates a config object and passes on *\*args* and *\*\*kwargs* to
        :meth:`.OrderedDict.__init__`.
        """

        OrderedDict.__init__(self, *args, **kwargs)

        # the original parsed code, used for saving things out
        self.__orig = {}

    @staticmethod
    def fromFile(filename, config=None, **kwargs):
        """Populates a config object from the file *filename*. If config
        is not given or ``None`` then a new, empty, config object is created to
        populate.

        :param filename: the file to load into the config object
        :param config: the config object to load the file into, if None a new
            config object will be created
        :type config: :class:`.Config`
        :param kwargs: Optional configuration options

            * overwrite - Overwrite existing values in the config object if one
              with the same name is found in the file to load.
            * group_on - the key to indicate new group so that new sub-config
              object can be created. When a kv-pair with the key name equal to
              the value of *group_on* is found a new config object is created and
              all proceededing kv-pairs are loading into that config object instead
            * groups_name - Used to group items into sub-config objects, the  new
              config object is created and placed into 
              ``getattr(self, self.groups_name)[new_conig.name] = new_config`` 
            * primary_group - the name of the group to copy its kv-pairs into
              this main config object

        Note: grouping is only handled one level deep. Groups cannot contain
        other groups.       
        """

        # overwrite existing values?
        overwrite = kwargs.pop('overwrite', False)

        # Config files can have grouped arguments
        # the variable to store in groups
        groups_name = kwargs.pop('groups_name', 'groups')
        # the name of the grouping key=value pair
        group_on = kwargs.pop('group_on', None)
        # the target group to extract
        primary_group = kwargs.pop('group', None)

        # If no config object was passed in, create one
        if config is not None:
            self = config
        else:
            self = Config(**kwargs)
        self._filename = filename
        self._path = os.path.abspath(os.path.dirname(filename))

        self[group_on] = primary_group

        # current group
        group = self  # start with the base config object as the group
        group_name = None
        groups = {}
        self[groups_name] = groups

        if filename is not None:
            file = open(filename, 'r')
            for line in file:
                line = line.strip()
                # skip comments
                if line == '' or line[0] in ('#', '%') or line[:2] in ('//',):
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # using eval() is inherently insecure, but allows for nice options
                # for setting options in the config file

                # first we attempt to evaluate the value without using the
                # config object as the locals
                no_locals_val = None
                try:
                    no_locals_val = eval(value)
                except:
                    pass

                # now we evaluate the value with the config object as the locals
                locals_val = None
                try:
                    locals_val = eval(value, {}, self.__dict__)
                except:
                    locals_val = value

                # if the key equals the group tag, start a new grouping
                if key == group_on:
                    group_name = locals_val
                    if group is not None:
                        self[locals_val] = group
                    group = Config(**kwargs)
                    groups[locals_val] = group

                    # start at the next line now that we have a group object
                    continue

                if type(locals_val) is str:
                    # try string replacement using the config object as the dict
                    try:
                        locals_val = locals_val % self
                    except KeyError:
                        pass
                    try:
                        locals_val = locals_val % group
                    except KeyError:
                        pass

                # if their string representations are not equal then the config
                # object, used as locals, was actually need to evaluate the value
                # so store the original string, it will be needed to reconstruct things
                if str(no_locals_val) != str(locals_val):
                    group.__orig[key] = value

                if overwrite:
                    group[key] = locals_val
                else:
                    cur_val = group.get(key, None)
                    group[key] = locals_val if cur_val is None else cur_val

                # if the current group is the target/primary group the add the
                # key=value directly to the config
                if group_name == primary_group:
                    if overwrite:
                        self[key] = locals_val
                    else:
                        cur_val = self.get(key, None)
                        self[key] = locals_val if cur_val is None else cur_val

            file.close()

            # if there is only one group, extract it outwards to the top level
            # if len(groups) == 1:
            # self.__dict__[group_on] = groups.iterkeys().next()
        return self


    def parseOpts(self):
        """Expands the --opts from a command line into key-value pairs and
        sets them on this config object. The values are attempted to be 
        coerced in the order: int, float, str.
        """

        for opt in self.opts:
            var, val = opt.split('=', 1)
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    # just a string
                    pass
            self[var] = val

    def saveTrees(self, trees, path=None, create=True, overwrite=True, return_path=False):
        """A helper method that is used to pickle a list of SRPTs or SRRFs to the
        location specified in *self.save_path* (which defaults to *path*).

        :param trees: the list of trees of forests to save to a pickle file
        :param path: the optional/default path to save the pickle to should 
            *self.save_path* not be set
        :param Boolean create: if the path doesn't exists, create the directories needed
        :param Boolean overwrite: if the file already exists, overwrite it
        :param Boolean return_path: instead of actually saving the trees, return
            the fully qualified filename they would be saved to. *This does not
            save/pickle anything!*

        :returns: If *return_path* is ``True`` return the filename it that the 
            trees/forests would be saved. Otherwise, return ``None``. 
        """

        path = self.get('save_path', path)

        if not path.endswith('/'):
            path += '/'

        if path is None:
            raise IOError('No path given for saving trees')

        # paths can reference things in the config object, so use string replacement now
        path = path % self
        path = self._getAbsolutePath(path)

        # make sure the path exists, and if it doesn't and we are supposed to,
        # create it
        if not os.path.exists(path):
            if create:
                os.makedirs(path)
            else:
                raise IOError('Path not found: %s' % path)

        # the format for the file name takes into account several options such as:
        #    * underlabel
        #    * num_trees
        # and customizes the file names based upon the existance (and value) of them
        if hasattr(self, 'underlabel'):
            pattern = 'run=%(run)s_numtrees=%(num_trees)s_samples=%(num_samples)s_underval=%(underval)s_underlabel=%(underlabel)s_pvalue=%(pvalue)s_depth=%(max_depth)s_distinctions=%(distinctions)s_stat=%(split_stat)s.pkl'

        elif hasattr(self, 'num_trees'):
            pattern = 'run=%(run)s_numtrees=%(num_trees)s_samples=%(num_samples)s_depth=%(max_depth)s_distinctions=%(distinctions)s_stat=%(split_stat)s.pkl'
        else:
            pattern = 'run=%(run)s_samples=%(num_samples)s_depth=%(max_depth)s_distinctions=%(distinctions)s_stat=%(split_stat)s.pkl'

        filename = os.path.join(path, pattern % self)

        # if we are just returning the path and not saving it
        if return_path:
            return filename

        if os.path.exists(filename) and not overwrite:
            raise RuntimeError('File exists and overwrite is False: %s' % filename)

        import pickle
        try:
            l = len(trees)
            print('Saving %s tree(s) to: %s' % (l, filename))
        except:
            print('Saving tree to: %s' % (filename))

        pickle.dump(trees, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)
        f_stat = os.stat(filename)
        print('\tsize:', formatBytes(f_stat.st_size))

    def getAsAbsolutePath(self, key, default=None):
        """Get the  value associated with *key* as an absolute path assuming the
        key is relative to the file this config object was loaded from.

        :param key: the key/attribute-name to use the associated value as a path
        :param default: the default value should *self[key]* not be set

        :returns: the absolute path of *self[key]*
        """

        filename = self.get(key, default)
        if filename is None:
            raise KeyError('Config key [%s] not found' % (key,))

        return self._getAbsolutePath(filename)

    def _getAbsolutePath(self, filename):
        """Helper function to get an absolute path of the *filename*,
        assuming filename is relative to the experiment file."""

        # find the correct path, in the experiment file they are either
        # relative to the experiment file, or an absolute path
        if filename != os.path.abspath(filename):
            return os.path.join(self._path, filename)
        else:
            return filename

    def _copy_up(self, obj, overwrite=False):
        """Copy the attributes from obj into this config object. Eqivalent of:

        .. code-block:: python

            for k, v in adict.__dict__.iteritems():
                setattr(config, k, v)

        :param obj: obj to copy attributes from
        :param Boolean overwrite: if an attribute with the same name already 
            exists in this config object, should it be overwritten?
        """

        for key, value in obj.__dict__.iteritems():
            if overwrite:
                self[key] = value
            else:
                cur_val = self.__dict__.get(key, None)
                self[key] = value if cur_val is None else cur_val

    def save(self, filename=None):
        """Save out the config object to the given *filename*, or use the original
        filename if it was loaded from one."""

        if filename is None:
            filename = self.__filename

        file = open(filename, 'w')
        for key in self:
            if key[:2] == '__':
                continue
            file.write(key)
            file.write(' = ')
            if key in self.__orig:
                value = self.__orig[key]
            else:
                value = self[key]
            file.write(str(value))

        file.close()

    def __getattr__(self, key):
        """Handle accessing the keys like attributes ``config['foo']`` is the 
        same as ``config.foo``.
        """

        if '__' not in key:
            if key in self:
                return self[key]
            else:
                return OrderedDict.__getattribute__(self, key)
        else:
            return OrderedDict.__getattribute__(self, key)

    def __setattr__(self, key, value):
        """Handle accessing the keys like attributes ``config['foo'] = 'bar'``
        is the same as ``config.foo = 'bar'``.
        """

        if '__' not in key:
            self[key] = value
        else:
            OrderedDict.__setattr__(self, key, value)

    def __reduce__(self):
        reduction = OrderedDict.__reduce__(self)

        # we dont want to pickle certain things
        no_pickle = ['graphs', 'experiments', 'distinction_generator', 'schema', 'datasets']
        if 'experiments' in self:
            no_pickle.extend(self.experiments.keys())
        items = [[k, self[k]] for k in self if k not in no_pickle]

        if len(reduction) == 2:
            reduction = (reduction[0], (items,))
        elif len(reduction) == 3:
            reduction = (reduction[0], (items,), reduction[2])

        return reduction

    def reload(self):
        """Reload the config object from an experiment file used to load it
        in the first place. Useful on unpickled-config objects since they
        don't pickle everything.
        """

        from experiment.ExperimentLoader import loadExperimentFile
        loadExperimentFile(self, self.exp)
        return self

    def __str__(self, **kwargs):
        if kwargs.get('pretty', False) == True:
            return 'Config[%s]' % '\n\t'.join(map(lambda kv: '%s=%s' % kv, self.items()))
        else:
            return 'Config[%s]' % (', '.join(map(lambda kv: '%r=%r' % kv, self.items())))

    def __repr__(self):
        return 'Config@%d[%s]' % (id(self), self.__dict__)
