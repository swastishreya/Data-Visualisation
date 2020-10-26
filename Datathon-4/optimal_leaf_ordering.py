import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
import itertools
import matplotlib.pyplot as plt
import seaborn
    
class OptimalLeafOrdering:
    def __init__(self, data, data_type='data', metric="euclidean", method='single'):
        
        self.data = data
        self.data_type = data_type
        self.metric = metric
        self.method = method

    def get_ordered_data(self):
        row_order = self.compute_dendrogram(axis=0)
        col_order = self.compute_dendrogram(axis=1)
        return self.data.iloc[row_order, col_order]
        
    def compute_dendrogram(self, axis=0):
        if axis == 1:
            data = self.data.T
        else:
            data = self.data

        # Calculate pairwise distances and linkage
        if self.data_type == 'data':
            pairwise_dists = distance.pdist(data.values, metric=self.metric)
        elif self.data_type == 'dist':
            pairwise_dists = []
            for i in range(len(data.values)):
                for j in range(i+1, len(data.values)):
                    pairwise_dists.append(data.values[i][j])
            pairwise_dists = np.array(pairwise_dists)
        else:
            raise NotImplementedError
        linkage = hierarchy.linkage(pairwise_dists, method=self.method)
        
        self.M = {}
        tree = hierarchy.to_tree(linkage)
        dists = distance.squareform(pairwise_dists)
        tree = self.order_tree(tree, dists)
        order = self.leaves(tree)
        del self.M
        return order
    
    def optimal_scores(self, v, D, fast=True):
        """ Implementation of Ziv-Bar-Joseph et al.'s leaf order algorithm
        v is a ClusterNode
        D is a distance matrix """

        def score(left, right, u, m, w, k):
            return get_M(left, u, m) + get_M(right, w, k) + D[m, k]

        def get_M(v, a, b):
            if a == b:
                self.M[v.get_id(), a, b] = 0
            return self.M[v.get_id(), a, b]

        if v.is_leaf():
            n = v.get_id()
            self.M[v.get_id(), n, n] = 0
            return 0
        else:
            L = self.leaves(v.left)
            R = self.leaves(v.right)
            LL = self.leaves(v.left.left, v.left)
            LR = self.leaves(v.left.right, v.left)
            RL = self.leaves(v.right.left, v.right)
            RR = self.leaves(v.right.right, v.right)
            for l in L:
                for r in R:
                    self.M[v.left.get_id(), l, r] = self.optimal_scores(v.left, D, fast=False)
                    self.M[v.right.get_id(), l, r] = self.optimal_scores(v.right, D, fast=False)
                    for u in L:
                        for w in R:
                            if fast:
                                m_order = sorted(self.other(u, LL, LR), key=lambda m: get_M(v.left, u, m))
                                k_order = sorted(self.other(w, RL, RR), key=lambda k: get_M(v.right, w, k))
                                C = min([D[m, k] for m in self.other(u, LL, LR) for k in self.other(w, RL, RR)])
                                Cmin = 1e10
                                for m in m_order:
                                    if self.M[v.left.get_id(), u, m] + self.M[v.right.get_id(), w, k_order[0]] + C >= Cmin:
                                        break
                                    for k in k_order:
                                        if self.M[v.left.get_id(), u, m] + self.M[v.right.get_id(), w, k] + C >= Cmin:
                                            break
                                        C = score(v.left, v.right, u, m, w, k)
                                        if C < Cmin:
                                            Cmin = C
                                self.M[v.get_id(), u, w] = self.M[v.get_id(), w, u] = Cmin
                            else:
                                self.M[v.get_id(), u, w] = self.M[v.get_id(), w, u] = \
                                    min([score(v.left, v.right, u, m, w, k) \
                                        for m in self.other(u, LL, LR) \
                                        for k in self.other(w, RL, RR)])
                    return self.M[v.get_id(), l, r]

    def order_tree(self, v, D, fM=None, fast=True):
        
        if fM is None:
            fM = 1
            self.optimal_scores(v, D, fast=fast)

        L = self.leaves(v.left)
        R = self.leaves(v.right)
        if len(L) and len(R):
            def getkey(z):
                u,w = z
                return self.M[v.get_id(),u,w]
            if len(L) and len(R):
                u, w = min(itertools.product(L,R), key=getkey)
            if w in self.leaves(v.right.left):
                v.right.right, v.right.left = v.right.left, v.right.right
            if u in self.leaves(v.left.right):
                v.left.left, v.left.right = v.left.right, v.left.left
            v.left = self.order_tree(v.left, D, fM)
            v.right = self.order_tree(v.right, D, fM)
        return v

    def other(self, x, V, W):
        # For an element x, returns the set that x isn't in        
        if x in V:
            return W
        else:
            return V

    def leaves(self, t, t2=None):
        """ Returns the leaves of a ClusterNode """
        try:
            return t.pre_order()
        except AttributeError:
            if t2 is not None:
                return t2.pre_order()
            else:
                return []
    
if __name__ == "__main__":
    # Create a staircase matrix
    X = np.zeros((100, 100))
    for n in [0,10,20,30,40,50,60]:
        X[int(10.*n/7):int(10.*(n+10)/7):,n:n+40] = 1

    X = distance.squareform(distance.pdist(X, metric="euclidean"))
    # X = distance.squareform(distance.pdist(X, metric="hamming"))

    seaborn.heatmap(X)
    plt.figure()

    # Since we know the data has a staircase pattern, we can now shuffle the rows and columns
    np.random.shuffle(X)
    X = X.T
    np.random.shuffle(X)
    X = X.T

    # Visualize the input data
    seaborn.heatmap(X)
    olo = OptimalLeafOrdering(pd.DataFrame(X), metric='hamming', method='complete')
    plt.figure()

    # Visualize the output data
    Y = olo.get_ordered_data()
    seaborn.heatmap(Y)
    plt.show()