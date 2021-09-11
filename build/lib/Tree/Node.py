"""
"""
from utilities.utils import instanceReducer


class Node(object):
    """
    Base class for all nodes in the SRPT tree it contains a fair bit of logic
    on how to actually construct trees
    """

    def __init__(self, parent):
        self.parent = parent
        self.config = parent.config
        self.depth = getattr(parent, 'depth', -1) + 1
        #print("Inside the node constructor:")
        #print("self.depth:%s"%self.depth)

        # not all nodes will populate this list but all have it
        self.children = []

        if self.__class__.__name__ == 'Node':
            # raise Exception()
            pass

    def __reduce__(self):
        return instanceReducer(self)

    def growTree(self, graphs,class_labels):
        """The guts of producing a SRPT, first we attempt to generate an internal
        node, if that succeeds we recursively tell that node to ``growTree()``.
        If generating an internal node fails (calling ``growTree()`` on the node
        returns ``None``), then we create a leaf node.

        :param graphs: the graphs to use as training data for growing the tree
        :type graphs: Dict[graph.id => STGraph]

        :returns: the grown tree or ``None`` if it wasn't possible to grow a tree
        :rtype: Subclass of :class:`srpt.Node.Node` or ``None``
        """

        self.depth -= 1
        #we are checking if the depth of the preceding node equal max_depth in order to get this
        #node as a constant probability node.
        if self.depth <= self.config.max_depth:
            # get a internal node (in a standard SRPT a distinction based node)
            node = self.getInternalNode(graphs)
            #print("node variable returned by Node.getInternalNode:")
            #print(node)
            #print("Type of node")
            #print(type(node))
        else:
            if self.config.get('verbose', False):
                print('|    ' * (self.depth + 1), 'Breaking due to maxdepth')
            node = None

        # try to grow the tree on that node, if it returns None then that means
        # the node couldn't grow so try for a leaf node

        if node is None or node.growTree(graphs,class_labels) is None:
            node = self.getLeafNode()
            node.growTree(graphs,class_labels)

        node.parent = self.parent
        node.depth = self.parent.depth

        return node

    def getInternalNode(self,graphs):
        """Returns an internal node, can be overridden for different types
        of trees.

        :returns: node that can be used as an internal node
        :rtype: Subclass of :class:`srpt.Node.Node`
        """
        from .DistinctionNode import DistinctionNode
        #####I need to remove this later on

        return DistinctionNode(self,graphs=graphs)

    def getLeafNode(self):
        """Returns a leaf node, can be overridden for different types of trees.
        Such as for different probability distributions at the leaf nodes.

        :returns: node that can be used as a leaf node
        :rtype: Subclass of :class:`srpt.Node.Node`
        """
        from .ConstantProbabilityNode import ConstantProbabilityNode

        return ConstantProbabilityNode(self)

    def __str__(self, depth=0, text='', simple=False):
        indent = '\t' * depth
        if text != '':
            if type(text) is str:
                lines = text.split('\n')
            else:
                lines = text
            text = ''
            for line in lines:
                text = '%s%s\t%s\n' % (text, indent, line)

        if not simple:
            s = '%s%s[depth=%s\n%s' % (indent, self.__class__.__name__, self.depth, text)
            for child in self.children:
                if child is None: continue

                s += '%s\n' % child.__str__(depth + 1)
            s += '%s]' % (indent)
        else:
            s = text
        return s