from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import math
import sys
#Used to perform statistical test and graph normal distributions
class Tester:

    def __init__(self, error1, error2 = None, e1_desc = '', e2_desc = ''):
        if not error1: return # Send Logic Error
        self.desc1 = e1_desc
        self.desc2 = e2_desc
        self.e1 = error1
        self.e2 = error2
        self.mean1 = self.calc_mean(self.e1)
        self.mean2 = self.calc_mean(self.e2)
        self.var1 = self.calc_variance(self.e1, self.mean1)
        self.var2 = self.calc_variance(self.e2, self.mean2)

    def calc_mean(self, error):
        if not error: return
        total_error = 0
        for e in error:
            total_error += e
        return total_error / len(error)

    def calc_variance(self, error, mean):
        if not error: return
        difference_sum = 0
        for e in error:
            difference_sum += ((e - mean)**2)
        return difference_sum / len(error)

    def get_mean(self):
        if not self.e2:
            return self.mean1
        return (self.mean1, self.mean2)

    def get_variance(self):
        if not self.e2:
            return self.var1
        return (self.var1, self.var2)

    def get_max(self):
        max_val = self.e1[0]
        for i in range(1, len(self.e1)):
            if max_val < self.e1[i]:
                max_val = self.e1[i]
        return max_val

    def get_min(self):
        min_val = self.e1[0]
        for i in range(1, len(self.e1)):
            if min_val > self.e1[i]:
                min_val = self.e1[i]
        return min_val

    def get_stdev(self, desc = None):
        if not desc: 
            if not self.var2: return math.sqrt(self.var1)
            return (math.sqrt(self.var1), math.sqrt(self.var2))
        if desc == self.desc1: return math.sqrt(self.var1)
        if desc == self.desc2 and self.var2: return math.sqrt(self.var2)
        else: return

    # Null hypothesis: There is no difference
    def compare(self):
        if not self.e2: return
        t = (self.mean1 - self.mean2) / math.sqrt((self.var1 / len(self.e1)) + (self.var2 / len(self.e2)))
        if len(self.e1) > len(self.e2):
            pval = 2 * stats.t.sf(abs(t), len(self.e2) - 1)
        else:
            pval = 2 * stats.t.sf(abs(t), len(self.e2) - 1)
        if pval > 0.05:
            self.output_comparison_results(("There was no statistically significant difference between %s and %s" % (self.desc1, self.desc2), pval)) # No significant difference
        else:
            self.output_comparison_results(("There was a statistically significant difference between %s and %s" % (self.desc1, self.desc2), pval)) # No significant difference


    def output_comparison_results(self, results = None):
        if not results: return # Send Logic Error
        print("%s with p-value: %s" % (results[0], results[1])) # Do something useful
        print("For %s we had mean %f and variance %f." % (self.desc1, self.mean1, self.var1))
        print("For %s we had mean %f and variance %f." % (self.desc2, self.mean2, self.var2))

    def graphResults(self):
        x1 = np.linspace(self.mean1 - 3 * math.sqrt(self.var1), self.mean1 + 3 * math.sqrt(self.var1), 100)
        x2 = np.linspace(self.mean2 - 3 * math.sqrt(self.var2), self.mean2 + 3 * math.sqrt(self.var2), 100)
        plt.plot(x1, mlab.normpdf(x1, self.mean1, math.sqrt(self.var1)))
        plt.plot(x2, mlab.normpdf(x2, self.mean2, math.sqrt(self.var2)))
        plt.show()
