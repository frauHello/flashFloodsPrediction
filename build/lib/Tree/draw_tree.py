"""
Uses graphviz to make pretty drawings of an SRPT or SRRF.
"""

import sys, os
from utilities.utils import argparser

try:
    import path_test  # @UnusedImport
except:
    sys.path.append(os.path.abspath('./'))

import pickle
from textwrap import wrap
from itertools import chain
import re
import pydot
import argparse

FORMATS = ['canon', 'cmap', 'cmapx', 'cmapx_np', 'dot', 'eps', 'fig', 'gd', 'gd2', 'gif', 'gv', 'imap', 'imap_np',
           'ismap', 'jpe', 'jpeg', 'jpg', 'pdf', 'plain', 'plain-ext', 'png', 'ps', 'ps2', 'svg', 'svgz', 'tk', 'vml',
           'vmlz', 'vrml', 'wbmp', 'xdot']






def main():

    path=r"C:\Users\user\Desktop\Prediction_model\experiment\flood\run=0_samples=100_depth=5_distinctions=all_stat=logrank.pkl"
    outfile=r"C:\Users\user\Desktop\Prediction_model\experiment\drawing.png"
    FILE = open(path, 'rb')
    all_trees = pickle.load(FILE)
    FILE.close()

    # how to get the label for an SRPTTree node
    def get_label(node):
        # leaf nodes have probabilities separated by commas, we don't want
        # to wrap them because will split split the lines on the commas ourselves
        label = node.__str__()
        ''.join(label)
        is_leaf = len(node.children) == 0
        if is_leaf:
            label="time steps: {0}, number of graphs: {1}".format(node.t,node.total)

        else:
            # label formatting
            name, body = label.split('[', 1)
            body = body.strip()[:-1]  # remove trailing bracket
            parts = re.split(r',\s+(\w+)\s*=', body)
            body = parts[0].strip()
            for i in range(1, len(parts), 2):
                body = '%s\\n%s=%s' % (body, parts[i].strip(), parts[i + 1].strip())
            label = '[%s]\\n%s' % (name.strip(), body)
            pass

        print(is_leaf)
        print(label)
        label = label.replace('\n', r'\n')
        return (label, not is_leaf)

    # how to get a label between parent and child in an SRPTTree
    def get_edge_label(parent, child):
        s = ''
        # first child is the 'Yes' branch, second child the 'No' branch
        if hasattr(parent, 'num_yes'):
            if child is parent.children[0]:
                # print repr(parent)
                s = 'Y (%s)' % (parent.num_yes)
                s += '\\nCounts[%s]' % (', '.join('%s=%s' % item for item in sortedItems(parent.split.num_yes_graphs)))
            elif child is parent.children[1]:
                s = 'N (%s)' % (parent.num_no)
                s += '\\nCounts[%s]' % (', '.join('%s=%s' % item for item in sortedItems(parent.split.num_no_graphs)))
        return s

    fname, ext = os.path.splitext(outfile)
    dot_outfile = '%s.dot' % (fname)
    outfile = '%s.%s' % (fname, ext[1:])


        # create the graph
    graph = buildTree(path,"",
                          has_children=lambda node: hasattr(node, 'children') and len(node.children) > 0,
                          get_children=lambda node: node.children,
                          get_label=get_label,
                          get_edge_label=get_edge_label)



    graph.write(dot_outfile, format='dot')
    graph.write(outfile, format=format)
    print('Tree picture written to:',outfile)


def sortedItems(adict):
    items = adict.items()
    items.sort()
    return items


def countNumYesAndNo(node, is_yes=False):
    num_yes = 0
    freq_yes = {}
    num_no = 0
    freq_no = {}
    if len(node.children) > 0:  # distinction node
        countNumYesAndNo(node.children[0], is_yes=True)
        num_yes += (node.children[0].num_yes + node.children[0].num_no)
        for label, count in chain(node.children[0].freq_yes.iteritems(), node.children[0].freq_no.iteritems()):
            freq_yes[label] = freq_yes.get(label, 0) + count

        countNumYesAndNo(node.children[1], is_yes=False)
        num_no += (node.children[1].num_yes + node.children[1].num_no)
        for label, count in chain(node.children[1].freq_yes.iteritems(), node.children[1].freq_no.iteritems()):
            freq_no[label] = freq_no.get(label, 0) + count
    else:  # leaf node
        if is_yes:
            num_yes = node.count
            freq_yes.update(node.frequency)
        else:
            num_no = node.count
            freq_no.update(node.frequency)
    node.num_yes = num_yes
    node.freq_yes = freq_yes
    node.num_no = num_no
    node.freq_no = freq_no


def buildTree(treename, root,
              has_children=lambda node: hasattr(node, 'children'),
              get_children=lambda node: node.children,
              get_label=str,
              get_edge_label=lambda: '',
              label_width=60):
    """Creates a dot graph file for use with visgraph's dot program. This works
    on many kinds of trees because of using has_children, get_children, and
    get_label which are user supplied functions for accessing information about
    nodes in the tree.

    has_children : should return True if the node has children, False otherwise
    get_children : should return a list of child nodes
    get_label    : should return the label for the node, defaults to str()
    label_width  : the number of columns to wrap the node labels to
    """

    graph = pydot.Dot()
    buildGraph(graph, root, has_children, get_children, get_label, get_edge_label, label_width)
    return graph


def buildGraph(graph, node, has_children, get_children, get_label, get_edge_label, label_width):
    # use the accessor function to get the label for the node
    node_label = get_label(node)

    # it can return a tuple where the first index is the label and
    # the second is a boolean indicating if wrapping should be done or not
    if type(node_label) is tuple:
        node_label, do_wrap = node_label
    else:
        do_wrap = True

    # quotes need to be escaped
    node_label = node_label.replace('"', '\\"')
    if do_wrap:
        # wrap returns a list, so join it into a single string, newlines must
        # be escaped also
        node_label = '\\n'.join(wrap(node_label, label_width))
    else:
        # escape the newlines
        node_label = node_label.replace('\n', '\\n"')

    # create a new node and add it to the graph
    gnode = pydot.Node(name=id(node), label=node_label)
    graph.add_node(gnode)

    # use accessor to check for children
    if has_children(node):
        # use accessor to get a list of children
        for child in get_children(node):
            # recursively call build graph on the children
            cnode = buildGraph(graph, child, has_children, get_children, get_label, get_edge_label, label_width)

            # add the edge, using the accessor to get the edge label
            edge = pydot.Edge(src=gnode, dst=cnode, label=get_edge_label(node, child))
            graph.add_edge(edge)

    return gnode


if __name__ == '__main__':
    main()
