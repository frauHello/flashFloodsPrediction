"""Provides the XMLObject class for simple creation of XML files
in a pythonic, oo way. An example is the simplest explanation.

Given this code::

    # create an XML object with the tag name 'Foo'
    foo = XMLObject('Foo')

    # creates a child text node with the tag name 'bar'
    foo.bar = 'this is a bar'

    # prefixing with underscores make element level parameters
    # <foo id='23'>
    foo._id = '23'

    # lists of other XMLObjects are written recursively, the parameter
    # name doesn't mater, in this case 'minions', it is just used in your
    # code, and doesn't show up in the XML
    foo.minions = [baz, car]

    foo.writeXML(open('foo.xml', 'w'))

Writes this XML::

    <Foo id='23'>
        <bar>this is a bar</bar>
        <baz>
            ...
        </baz>
        <car>
            ...
        </car>
    </Foo>

Since keeping track of child objects is a very common thing to do, XMLObject
provides a bit of convenience to pretty up code::

    # make a parent XML object
    parent = XMLObject('parent')
    # and a list to hold children
    parent.children = []

    # Instead of doing this:
    child = XMLObject('child')
    parent.children.append(child)

    # You can simply do:
    child = XMLObject('child', parent.children)

XML Object also allows finer grained control when memory is an issue. Since
creating a large document entirely in memory can cause out-of-memory errors, the
solution is to write the document in chunks. This is done using writeStart() and
writeEnd(). Below is an example::

    # file handle to xml file
    fh =  ...

    # an xml object with lots of children
    big_parent = XMLObject('big_parent')
    big_parent._id = 8
    big_parent.color = 'red'

    # write out the children individually
    # first: write out the start of big_parent, we have to tell XMLObject that
    # we will be writing children manually.
    big_parent.writeStart(fh, has_children=True)

    # write out each child XMLObject separately
    for child_idx in xrange(num_children):
        # create an child XMLObject that consumes to much memory to keep every
        # child in memory
        child = complicatedChildMakingFunction(child_idx)

        # in order to format the XML properly we have to tell it what indention
        # depth to write the child at.
        child.writeXML(fh, depth=1)

        # "child" goes out of scope now and can be garbage collected

    # write the end of the big_parent to finish things up
    big_parent.writeEnd(fh)

"""

import sys, os

try:
    import path_test
except:
    sys.path.append(os.path.abspath('./'))


import numpy as np
from io import StringIO


def main():
    # simple test
    import pickle
    o = XMLObject('foo')
    o._id = 1
    o.foo = 'bar'
    #print(o)

    p = pickle.dumps(o)
    #print(p)


    o2 = pickle.loads(p)
    #print('PrePickle:\n', o)
    #print('PostPickle:\n', o2)


class XMLObject(object):
    """A simple object that knows how to right itself out as XML. Introspection
    is used in order to decide what should be written out. Performs fast writing
    of numpy arrays, so generally leave numpy arrays alone, don't convert to a
    list. See the module doc string for example usage.

    Note: The xml elements are written in the same order that they
    were set on the XMLObject.
    """

    def __init__(self, name, parent=None, **kwargs):
        """Create an XML object with the tag-name 'name', optionally appends the
        new object to a list, and updates the object with the parameters
        in \*\*kwargs (i.e. self.key = value for all key-value pairs).

        :param name: The tag-name of the object
        :param parent: An optional list to append this object too
        :param \*\*kwargs: Keyword arguments which are set on this object.
        """

        self.__name = name
        self.__order = []
        self.__children = []
        self.__tag_attributes = []

        if parent is not None:
            if hasattr(parent, 'append'):
                parent.append(self)
            elif hasattr(parent, '__setitem__'):
                parent[name] = self

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def clone(self, name=None):
        """Shallow clone of the XMLObject. Optional renames the object.

        :param name: Optional new tag-name of the cloned object.

        :returns: A clone of this object
        """

        my_name = name if name is not None else self.__name
        a_clone = XMLObject(my_name)

        for key in self.__order:
            setattr(a_clone, key, self.__dict__[key])

        return a_clone

    def __setitem__(self, name, value):
        """Allows using either dictionary syntax or dotted attribute syntax
        for setting attributes on an XML object::
            obj['your_attr'] = v
        Is equivalent to::
            obj.your_attr = v
        """

        self.__setattr__(name, value)

    def __setattr__(self, name, value):
        """Allows using either dictionary syntax or dotted attribute syntax
        for setting attributes on an XML object::
            obj['your_attr'] = v
        Is equivalent to::
            obj.your_attr = v
        """

        # we need to keep track of the order things were set in
        # so that the xml is written in the expected order

        # our own dict provides quick checking to see if the attrib exists
        # this is faster then checking if the attrib is in the ordered list
        exists = name in self.__dict__
        object.__setattr__(self, name, value)

        # don't track special attribs '__*__' and make sure its not already
        # in the list of insertion/set order
        if '__' not in name and not exists:
            # print '%s.%s = %s' % (self.__name, name, value)
            self.__order.append(name)

    def remove(self, *attributes):
        """Remove the given attributes from the XMLObject.
        E.g. obj.remove('foo', 'bar', 'baz')

        :param \*attributes: Attributes to remove"""

        my_dict = self.__dict__
        if len(attributes) == 1:
            try:
                if not isinstance(attributes[0], str):
                    attributes = tuple(attributes[0])
            except:
                pass

        # clean up the objects __dict__ and the list that
        # tracks the order attributes were set
        for field in attributes:
            my_dict.pop(field, None)
        self.__order = [field for field in self.__order if field not in attributes]

        return self

    def writeStart(self, fh, depth=0, haschildren=False):
        """Helper function that writes the beginning of the XMLObject.

        Can be called by the user for finer control over xml writing by starting
        the xml object and forcing the write to the file. Then the user in responsible
        for writing all children explicitly.


        :param fh: File handle to write the XML to
        :param depth=0: The identation level to write the object at
        :param haschildren=False: Will this object have children (which must be explicitly written out.)
        """

        my_dict = self.__dict__
        indent = '\t' * depth  # track the indention level

        # write out the beginning of the tag: <foo
        fh.write('%s<%s' % (indent, self.__name))

        self.__tag_attributes = []
        # filter the object parameters into those that are tag attributes, and those that aren't
        for k in self.__order:
            # tag attributes start with underscores '_'
            if k[0] == '_' and k.lower() != '_text' and '__' not in k:
                self.__tag_attributes.append(k)

        # write out the tag attributes
        for child in self.__tag_attributes:
            # write out the tag element: id='1'
            value = my_dict[child]
            value = self._format(value, indent)
            fh.write(' %s=\'%s\'' % (child[1:], value))

        # close the beginning tag: >'

        self.__haschildren = haschildren
        if haschildren:
            fh.write('>\n')

    def writeXML(self, fh, depth=0):
        """Given a file handle, write out the XML of this object and all children recursively.

        :param fh: File handle to write the XML to
        :param depth=0: The identation level to write the object at
        """

        # <foo id='3'> -- id='3' is called an tag attribute in this code
        #    <bar>...</bar> -- a child of foo
        # </foo>
        my_dict = self.__dict__

        self.__children = []
        # filter the object parameters into those that are tag attributes, and those that aren't
        for k in self.__order:
            # tag attributes start with underscores '_'
            if k[0] != '_' and '__' not in k:
                self.__children.append(k)
            elif k == '_text':
                self.__children.append(k)

        # write the beginning/opening of the xml element
        self.writeStart(fh, depth, len(self.__children) > 0)

        if len(self.__children) == 0:
            self.writeEnd(fh, depth)
        else:
            # fh.write('\n')

            # write out the children of the tag
            for child in self.__children:
                value = my_dict[child]

                # the text element is handled specially
                # <foo>this is some text</foo> -- "this is some text" is a text element
                if child.lower() == '_text':
                    self._writePlain(fh, child, value, depth + 1, is_text_element=True)

                # if it is a list-type
                elif isinstance(value, (list, tuple, np.ndarray, dict)):
                    if isinstance(value, dict):
                        value = value.values()

                    # check if its uniformly a list of primitives, if so pass it straight
                    # on to writePlain
                    uniform, val_type = self._checkUniform(value)
                    if uniform and val_type is not XMLObject:
                        self._writePlain(fh, child, value, depth + 1)
                    else:
                        for item in value:
                            if isinstance(item, XMLObject):
                                item.writeXML(fh, depth + 1)
                            else:
                                self._writePlain(fh, child, item, depth + 1)

                # if the child is an XML object recursively write it out
                elif isinstance(value, XMLObject):
                    value.writeXML(fh, depth + 1)

                elif value is None:
                    pass  # skip none-values

                else:
                    self._writePlain(fh, child, value, depth + 1)
            self.writeEnd(fh, depth)

    def writeEnd(self, fh, depth=0):
        """Writes the end of the XMLObject. See :meth:`~.writeStart`.

        :param fh: File handle to write the XML to
        :param depth=0: The indentation level to write the object at
        """

        if len(self.__children) > 0 or self.__haschildren:
            indent = '\t' * depth  # track the indention level
            fh.write('%s</%s>\n' % (indent, self.__name))
        else:
            fh.write(' />\n')

    def _writePlain(self, fh, name, value, depth, is_text_element=False):
        """Helper function for writing an XML element out to the file.

        :param fh: File handel to write the XML to
        :param name: Tag-name of the XML tag
        :param value: The value that gets written between the start and end tag
        :param depth: The depth to write the object at
        :param is_text_element=False: Is this a text element of another tag?
        """

        indent = '\t' * depth  # track the indentation level

        # if its a list the join it with commas
        value = self._format(value, indent, replace_new_lines=is_text_element)

        if is_text_element:
            fh.write('%s%s\n' % (indent, value))
        else:
            fh.write('%s<%s>%s</%s>\n' % (indent, name, value, name))

    def _checkUniform(self, iterable):
        """Checks if an iterable is recursively uniformly of the same type.

        :param iterable: The iterable to check

        :returns: True if the iterable is all the same type."""

        # numpy arrays are uniform by requirement
        if isinstance(iterable, np.ndarray):
            return True, str(iterable.dtype)

        # check each item in the iterable to ensure a consistent type
        old_type = None
        cur_type = None
        for v in iterable:
            cur_type = type(v)
            if old_type is None:
                # initial item in list
                continue

            if old_type != cur_type:
                # print 'NotUnifrom:', iterable
                return False

            try:
                # if the item is iterable check that its uniform, if its not uniform
                # then by extension this iterable isn't either
                if not self._checkUniform(iter(v))[0]:
                    # print 'NotUnifrom:', iterable
                    return False
            except TypeError:
                pass

            old_type = cur_type

        return True, cur_type

    def _format(self, value, indent, replace_new_lines=False):
        """Helper function for formatting objects to be written as XML.

        :param value: The value to format into acceptable XML
        :param indent: The string to use as indention
        :param replace_new_lines: True if new lines in values of type "str"
            should be replaced with: "%s\n" % indent
        """

        # For lists (including nested lists),
        #            we want the deepest nesting to be ',' joined,
        #        the next to be ';', joined;
        #    and the next to be '|' joined
        all_joiners = (',', ';', '|')
        stringio = StringIO()

        # keeps track of what joiner to use for the current nesting
        joiners = list()

        # internal formating function that is called recursively on lists to
        # write them to XML. does fast numpy array writing using numpy.savetxt
        def format(v, depth=0):
            # recursively write tuples, lists, and numpy arrays
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v) == 0:
                    v = ""
                else:
                    if len(joiners) == depth:
                        joiners.append(all_joiners[depth])

                        # fast writing of numpy-arrays by letting numpy do the work
                    if isinstance(v, np.ndarray):
                        # performs a reshape to 1D without memory copy by
                        # returing a new view of the array
                        v = v.ravel().reshape(1, -1)

                        # get the joiner to use
                        joiner = joiners[-(depth + 1)]

                        # clear the buffer
                        stringio.truncate(0)

                        # format integers without decimal points (makes for much
                        # nicer XML)
                        if issubclass(v.dtype.type, np.integer):
                            np.savetxt(stringio, v, fmt='%i', delimiter=joiner)

                        # format floating point to 6 significant digits
                        else:
                            np.savetxt(stringio, v, fmt='%.6f', delimiter=joiner)

                        v = stringio.getvalue()[:-1]  # remove newline appened by numpy

                    # write lists or tuples
                    else:
                        v = map(lambda x: format(x, depth + 1), v)

                        joiner = joiners[-(depth + 1)]
                        v = joiner.join(v)

            # write all other types of objects
            else:
                # write using str() to convert object to text, and reformat
                # newlines to be indented properly
                if isinstance(v, str) and replace_new_lines:
                    v = v.replace('\n', '\n%s' % indent)
                else:
                    v = str(v)

            return v

        return format(value)

    def __delattr__(self, name):
        """Delete an attribute. Note, this removes the insertion ordering, so
        if you later set the value agian, it will not be in the former position.
        """

        if '__' in name:
            object.__delattr__(self, name)
        else:
            # we call our own remove function to clean up our internal data
            # structures
            self.remove(name)

    def __str__(self):
        """Convert the XMLObject into a printable string."""

        my_name = self.__name
        my_items = []
        for k, v in self.__dict__.items():
            if '__' not in k:
                my_items.append((k, v))
        return 'XMLObject(%s)[%s]' % (my_name, ', '.join(map(lambda k_v: '%s=%s' % k_v, my_items)))


if __name__ == '__main__':
    main()
