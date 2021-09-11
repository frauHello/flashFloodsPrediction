import sys

from collections import defaultdict
from itertools import chain
import string

from math import log
import numpy as np

try:
    import scipy.stats
except ImportError:
    print('Missing scipy.stats, functionality will be limited')

from utilities.utils import instanceReducer


class ContingencyTable(object):
    """Represents a contingency table which is structured with rows being the 
    actual class and columns being the predicted class. Hence, table[i][j] is 
    the number of times a graph with the real class *i* was predicted, to be 
    class *j*.

    Instead of creating one directly it is generally done by calling
    :func:`.fromLabeling` which uses a standard labeling returned by the 
    various *labelGraphs()* functions. 
    """

    _init_args = ('table', 'row_header', 'column_header', 'stat')

    def __init__(self, table, row_header='', column_header='', stat='chi2'):
        """Create a contingency table

        :param table: either a ndarray (or 2d list/tuple) where table[i][j] is 
            the number of times class i was predicted, to be class j, or a 
            nested Dict[real_class => Dict[predicted_class => counts]]  
        :param row_header: the name to give the rows of the table
        :param column_header: the name to give the columns of the table
        :param stat: the statistic used to calculate the *pvalue* and *fit* of
            the table. Can be one of:

                * chi2 - Chi-Squared
                * sgnchi2 - Signed Chi-Squared
                * g - G Statistic
                * tss - True Skill Score
                * gini - Gini Statistic
                * igr - Information Gain Ratio
        """

        self.row_header = row_header
        self.column_header = column_header
        self.row_names = None
        self.col_names = None

        if type(table) in (dict, defaultdict):
            raw_table = table

            # convert from a dict table to a normal table
            # Dict[RealLabel] => Dict[PredictedLabel] => counts
            row_names = list(raw_table.keys())
            num_rows = len(row_names)
            row_names.sort()

            col_names = list(raw_table[row_names[0]].keys())
            num_cols = len(col_names)
            col_names.sort()

            table = np.zeros((num_rows, num_cols))
            for r, row in enumerate(row_names):
                for c, col in enumerate(col_names):
                    table[r][c] = raw_table[row][col]

            self.row_names = map(str, row_names)
            self.col_names = map(str, col_names)

        elif type(table) in (list, tuple):
            table = np.array(table, dtype=float)

        if self.row_names is None:
            self.row_names = map(str, range(table.shape[0]))
            self.col_names = map(str, range(table.shape[1]))

        # calc some basic info about the table
        self.stat = stat
        self.table = table
        self.observed = table
        self.num_rows = table.shape[0]
        self.num_cols = table.shape[1]
        self.col_margins = table.sum(axis=0)
        self.row_margins = table.sum(axis=1)
        self.total = table.sum(axis=None)

        self.expected = np.outer(self.row_margins, self.col_margins) / self.total
        self.fit=np.empty(self.expected.shape)
        # calculate a bunch of statistics
        self._calcTSS()
        self._calcHSS()
        self._calcMCC()
        self._calcGSS()

        if stat == 'chi2':
            self.dof = (self.num_cols - 1) * (self.num_rows - 1)
            for i in range(self.expected.shape[0]):
                for j in range(self.expected.shape[1]):
                    if self.expected[i][j]==0.0:
                        self.fit[i][j]=0.0
                    else:
                        self.fit[i][j]=((self.observed[i][j] - self.expected[i][j]) ** 2) / self.expected[i][j]
            ## self.fit = (((self.observed - self.expected) ** 2) / self.expected)
            self.fit = self.fit.sum(axis=None)
            self.pvalue = 1.0 - scipy.stats.chi2.cdf(self.fit, self.dof)

        elif stat == 'g':
            self.dof = (self.num_cols - 1) * (self.num_rows - 1)
            mask = self.observed > 0
            self.fit = 2.0 * self.observed[mask] * np.log(self.observed[mask] / self.expected[mask])
            self.fit = sum(self.fit)
            self.pvalue = 1.0 - scipy.stats.chi2.cdf(self.fit, self.dof)

        elif stat == 'sgnchi2':
            self.dof = (self.num_cols - 1) * (self.num_rows - 1)
            self.fit = ((self.observed - self.expected) ** 2) / self.expected
            self.fit[(self.observed < self.expected)] *= -1
            self.fit = sum(self.fit)
            self.pvalue = 1.0 - scipy.stats.chi2.cdf(self.fit, self.dof)

        elif stat == 'tss':
            self.fit = self.tss
            dof = (self.num_cols - 1) * (self.num_rows - 1)
            chi2 = (((self.observed - self.expected) ** 2) / self.expected).sum(axis=None)
            self.pvalue = 1.0 - scipy.stats.chi2.cdf(chi2, dof)

        elif stat == 'gini':
            # gini impurity of the root
            i_t = 1 - ((self.col_margins / self.total) ** 2).sum()

            # gini impurity for the branches
            i_yes = 1 - (self.table[0] / self.row_margins[0] ** 2).sum()
            i_no = 1 - (self.table[1] / self.row_margins[1] ** 2).sum()

            # relative class frequencies in the branches
            p_yes, p_no = self.row_margins / self.total

            # change in gini importance caused by the split
            delta_gini = i_t - p_yes * i_yes - p_no - i_no
            self.fit = delta_gini

        elif stat == 'igr':
            def I(S):
                i = 0
                s = S.sum()
                if s == 0: return 0
                for c in S:
                    if c == 0:
                        continue

                    rf = (c / s)
                    i -= rf * log(rf, 2)
                return i

            Gsb = I(self.col_margins.astype(float))
            Psb = 0
            for i in range(self.num_rows):
                Si = self.table[i].astype(float)
                sz_Si = Si.sum(axis=None)
                if sz_Si == 0:
                    continue

                Gsb -= (sz_Si / self.total) * I(Si)
                Psb -= (sz_Si / self.total) * log(sz_Si / self.total, 2)
            self.fit = Gsb / Psb
            if np.isnan(Psb):
                self.fit = 0

            # Quinlan suggests limiting splits to those with average information
            # gain, so only pick splits with Gsb >= .5, since the srpt code
            # checks for pvalues < threshold invert Gsb so that  a Gsb of >= .5
            # yields a "pvalue" <= .5.
            self.pvalue = 1. - Gsb


        elif stat == 'exp2':
            dof = (self.num_cols - 1) * (self.num_rows - 1)
            d1 = np.diag(table).astype(float)
            table2 = np.fliplr(table)
            d2 = np.diag(table2).astype(float)

            if d1.sum() < d2.sum():
                d1, d2 = d2, d1
                table = table2
            table = table.astype(float) / self.total
            d1 /= self.total
            print(table)
            print('   diag:', d1)
            print('colsums:', table.sum(axis=0))
            print('colsums:', table.sum(axis=0) - d1)
            fit = (d1 / (table.sum(axis=0) - d1))
            print('fit:', fit)
            self.fit = fit.sum() / (d1.sum() * self.total)
            self.pvalue = (fit - fit.mean()).std()

        # ensure its not a numpy.float dtype
        self.fit = np.float(self.fit)
        self.pvalue = np.float(self.pvalue)

    def __reduce__(self):
        return instanceReducer(self)

    @staticmethod
    def getMinObservations(stat):
        """Minimum number of items in a table for the results of the fitting
        statistic to be meaningful.

        :param stat: fitting statistic

        :returns: number of items
        """
        #print("Inside getMinObservations:")
        if stat == 'chi2' or stat == 'sgnchi2':
            return 20
        elif stat == 'igr':
            return 10
        elif stat=='logrank':
            return 2
        else:
            return 20

        #raise ValueError("Unknown statistic: %s" % stat)

    def _calcTSS(self):
        """Calculates the TSS (True Skill Score) of this table."""

        self.tss = None
        if (self.num_cols == self.num_rows):
            A = self.table.trace() / self.total
            B = np.dot(self.row_margins, self.col_margins) / (self.total ** 2)
            C = np.dot(self.row_margins, self.row_margins) / (self.total ** 2)
            numerator = A - B
            denominator = 1 - C

            # test for being arbitrarily close to zero
            EPSILON = 1e-5
            if (abs(numerator) < EPSILON or abs(denominator) < EPSILON):
                self.tss = 0.
            else:
                self.tss = float(numerator / denominator)

        return self.tss

    def _calcHSS(self):
        """Calculate Heidke Skill Score."""

        self.hss = None
        if (self.num_cols == self.num_rows):
            A = self.table.trace() / self.total
            B = np.dot(self.row_margins, self.col_margins) / (self.total ** 2)
            self.hss = (A - B) / (1 - B)
        return self.hss

    def _calcGSS(self, return_scoring=False):
        """Calculate Gerrity Skill Score."""

        self.gss = None
        if (self.num_cols != self.num_rows):
            return self.gss

        #
        # This an implementation of calculating GSS as taken from Gerrity's
        # paper "A Note on Gandin and Murphy's Equitable Skill Score"
        # http://http://adsabs.harvard.edu/abs/1992MWRv..120.2709G
        #

        # probability distribution of the actual class labels
        real_class_prob = self.row_margins / float(self.total)

        # Defined as per Gerrity
        def D(n):
            subset_sum = real_class_prob[:n].sum()
            d = (1. - subset_sum) / subset_sum
            return d

        # Defined as per Gerrity
        def R(n):
            r = 1. / D(n)
            return r

        K = len(real_class_prob)  # number of classes

        # constant used by Gerrity
        k = 1. / (K - 1)

        # empty score matrix to fill
        S = np.zeros((K, K), dtype=float)

        # convience function to sum over the function f(x) over x where x is
        # from the set {n, n+1, n+2, ..., m-1, m}
        def _sum(f, n, m):
            return sum(map(f, range(n, m + 1)))

        # value of the matrix on the diagonal
        def Snn(n):
            x = _sum(R, 1, n - 1)
            y = _sum(D, n, K - 1)
            s = k * (x + y)
            return s

        # value of the matrix off the diagonal
        def Smn(m, n):
            if m >= n: m, n = n, m
            x = _sum(R, 1, m - 1)
            y = _sum(lambda n: -1, m, n - 1)
            z = _sum(D, n, K - 1)

            s = k * (x + y + z)
            return s

        # conversion of the contingency table (a CDF) into a PDF
        E = (self.table / float(self.total))

        # calculate the value of each cell of the matrix, not the offsets
        # for python's 0-based arrays and the mathematical formula being 1-based
        for m in range(1, K + 1):
            for n in range(1, K + 1):
                if m == n:
                    s = Snn(n)
                else:
                    s = Smn(m, n)
                S[m - 1, n - 1] = s

        # final GSS calculate using the scoring matrix
        self.gss = np.dot(S.T, E).trace()

        if return_scoring:
            return self.gss, E, S
        else:
            return self.gss

    def _calcMCC(self):
        """Matthews Correlation Coefficient (aka Phi Coefficient)::

            sqrt(chi^2 / total)

        """

        self.mcc = None
        if (self.num_cols == self.num_rows):
            chi2 = (((self.observed - self.expected) ** 2) / self.expected).sum(axis=None)
            self.mcc = np.sqrt(chi2 / self.total)
        return self.mcc

    @staticmethod
    def fromLabeling(class_labels, labeling):
        """Given a labeling from calling labelGraphs() on a tree, create a
        contingency table. 

        :param config: the config object with all experiment information. whats really
            needed is the list of class labels form *config.schema.class_labels*
        :type config: :class:`~utilities.Config.Config`
        :param labeling: the labeling to create the table from
        :type labeling: Dict[STGraph => (class_label, PDF)].

        :returns: the contingency table
        :rtype: :class:`.ContingencyTable`
        """

        # setup the contingency table by initializing the counts for all combinations
        # of real labels vs guessed labels
        # Dict[RealLabel] => Dict[PredictedLabel] => counts
        class_table = {}
        for real_label in (str(lbl) for lbl in class_labels):
            class_table[real_label] = {}
            for guessed_label in (str(lbl) for lbl in class_labels):
                class_table[real_label][guessed_label] = 0
                class_table[real_label][guessed_label] = 0

        # for each graph in the labeling create the confusion table indexed
        # by row,column=>real,guessed/predicted
        # Dict[RealLabel] => Dict[PredictedLabel] => counts
        for graph, label in labeling.items():
            label = label[0]
            try:
                print("graph.class_label: %s"%graph.class_label)
                print("type:graph.class_label: %s" % type(graph.class_label))
                print("label: %s" % label)
                print("type:label: %s" % type(label))
                class_table[graph.class_label][label] += 1
            except:
                print('real:%s, guess:%s' % (graph.class_label, label))

                repr(graph)
                raise

        # now that we have the table create a ContingencyTable object and return it
        contable = ContingencyTable(class_table, row_header='Real', column_header='Predicted')
        return contable

    def __str__(self):
        """Pretty print the table."""
        # find the longest column name
        max_colname_len = 0
        for name in chain(self.row_names, self.col_names):
            max_colname_len = max(max_colname_len, len(name))

        # find the longest value when converted into a string
        max_value_len = max_colname_len
        for row in self.table:
            for col in row:
                max_value_len = max(max_value_len, len(str(col)))
        max_value_len += 2

        # we indent the table 4-spaces if there is no row_header
        indent = ' ' * 4
        if self.row_header is not None and self.row_header != '':
            indent = ' ' * (len(self.row_header) + 2)
        else:
            self.row_header = ' ' * 2  # makes up the indentation on the first row
        print("type(self.col_names):")
        print(type(self.col_names))
        # start the table and write the columns header
        s = 'Table[%s%s\n' % (' ' * (
        max_colname_len + len(indent) + 4 - 6 + int(.5 * len(list(self.col_names)) * (max_value_len + 2)) - int(
            .5 * len(self.column_header))), self.column_header)

        # indent before the column names
        s += indent + ' ' * (max_colname_len + 4)
        # write the column names
        for name in self.col_names:
            s += '[%s]' % string.center(name, max_value_len)
        s += '\n'

        # write each row of the table
        for r, row in enumerate(self.row_names):
            # the very first row will get the rows header, other rows are just indented
            row_indent = indent if r > 0 else ' %s ' % self.row_header

            # write the row name
            s += ('%s[ %%%ds ]' % (row_indent, max_colname_len)) % row

            # write the values
            for c, col in enumerate(self.col_names):
                s += '[%s]' % string.center(str(self.table[r][c]), max_value_len)
            s += '\n'
        s += ']'
        return s


def calcAUCFromLabeling(config, labeling):
    """Calculates the AUC of a labeling returned by
    :meth:`SRPTree.labelGraphs <srpt.SRPTree.SRPTree.labelGraphs>` or
    :meth:`SRRForest.labelGraphs <srrf.SRRForest.SRRForest.labelGraphs>`

    :param config: the config object with all experiment information. whats really
        needed is the list of class labels form *config.schema.class_labels*
    :type config: :class:`~utilities.Config.Config`
    :param labeling: a labeling
    :type labeling: Dict[STGraph => label]
    """

    # normal AUC for two classes
    if len(config.schema.class_labels) == 2:
        target_class = config.schema.class_labels[1]
        return _calcAUCFromLabeling(config, labeling, target_class)

    # give each class a chance at being the "positive" class and average them
    aucs = []
    for target_class in config.schema.class_labels:
        aucs.append(_calcAUCFromLabeling(config, labeling, target_class))
    return float(np.mean(aucs))


def _calcAUCFromLabeling(config, labeling, target_class):
    """Utlitiy function for calculating the AUC from a labeling for
    a given target class.

    Note: This is largely converted from several other implementations of
    calculating AUC."""

    probs = []
    labels = []
    for graph, label_prob in labeling.items():
        probs.append(label_prob[1].get(target_class, 0))
        labels.append(1 if graph.class_label == target_class else 0)

    idx = range(len(labels))
    sorted(idx,key=lambda i: probs[i])
    #idx.sort(key=lambda i: probs[i])

    labels = np.array(labels)[idx]
    probs = np.array(probs)[idx]

    # calculate the number of positive/negative instances
    neg_idxs = np.where(labels == 0)[0];
    pos_idxs = np.where(labels == 1)[0];

    num_neg = float(len(neg_idxs));
    num_pos = float(len(pos_idxs));

    all_probs = [0]
    all_probs.extend(probs)
    all_probs.append(1.1);

    false_pos_ratio = []
    true_pos_ratio = []
    # now loop through each of the probabilities and calculate the ratios
    for prob in all_probs:
        # calculate the number and ratio of false positives
        # by examining the negative instances
        num_false_pos = len(np.where(probs[neg_idxs] >= prob)[0]);
        if num_neg > 0:
            ratio = num_false_pos / num_neg
        else:
            ratio = 0
        false_pos_ratio.append(ratio);

        # calculate the number and ratio of true positives
        # by examining the positive instances
        num_true_pos = len(np.where(probs[pos_idxs] >= prob)[0]);
        if num_pos > 0:
            ratio = num_true_pos / num_pos
        else:
            ratio = 0
        true_pos_ratio.append(ratio);

    # calculate the area under the curve using trapezoidal approximation
    auc = 0
    for i in range(len(false_pos_ratio) - 1):
        # x1 > x2, y1 > y2
        x1 = false_pos_ratio[i]
        x2 = false_pos_ratio[i + 1]
        y1 = true_pos_ratio[i]
        y2 = true_pos_ratio[i + 1]

        w = x1 - x2
        h1 = y2
        rect = (w * h1)

        h2 = y1 - y2
        tri = (w * h2) / 2.

        auc += (rect + tri)
    return auc


def main():
    #    table = {0:{0:15, 1:4}, 1:{0:232, 1:250}}
    #    tbl = ContingencyTable(table, row_header='Real', column_header='Guessed')
    #    print tbl
    #    print 'Pvl:', tbl.pvalue
    #    print '%s: %s' % (tbl.stat, tbl.fit)
    #    print 'Tss:',tbl.tss
    #
    #    tbl = ContingencyTable(table, row_header='Real', column_header='Guessed', stat='exp')
    #    print
    #    print tbl
    #    print 'Pvl:', tbl.pvalue
    #    print '%s: %s' % (tbl.stat, tbl.fit)
    #    print 'Tss:',tbl.tss
    #
    #    # Table 2.11 (p36) from Cohen's book
    #    table = np.array([[30,5],
    #                      [32,8],
    #                      [53,16]], dtype=float)
    #
    #    # chi squared given in book
    #    cohen_chi2 = 1.145
    #
    #    contbl = ContingencyTable(table, stat='chi2')
    #    print 'Expected Chi2:', cohen_chi2
    #    print 'Calculated Chi2:', contbl.fit
    #    print 'Pvalue:', contbl.pvalue
    #    print
    #table = ContingencyTable(np.array(((169, 263), (62, 1150))))
    #sys.exit()

    tables = ([[12, 0], [0, 23]],
              [[34, 1], [3, 23]],
              [[0, 12], [23, 0]],
              [[23, 0], [0, 12]],
              [[12, 13], [14, 12]],
              [[52, 13], [14, 32]],
              [[50, 36], [2, 12]],
              [[25, 18], [1, 6]],
              [[0, 0], [5, 5]],
              [[2, 0], [171, 576]],
              [[172, 1], [1, 575]],
              )

    stats = ['chi2', 'igr']
    for table in tables:
        table = np.array(table)

        print(table)
        for stat in stats:
            tbl = ContingencyTable(table, stat=stat)
            print('%4s=%6.3F, p=%0.4F' % (stat, tbl.fit, tbl.pvalue))


if __name__ == '__main__':
    main()
