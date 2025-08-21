import numpy as np
from scipy.spatial import cKDTree

def twonn_id(X, k=2):
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=k+1)
    r1 = dists[:,1]
    r2 = dists[:,2]
    ratios = r2 / (r1 + 1e-12)
    return float(1.0 / (np.mean(np.log(ratios + 1e-12)) + 1e-12))
