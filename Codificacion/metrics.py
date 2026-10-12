import numpy as np
from scipy.spatial.distance import hamming
def euclideanDistance(v1,v2):
    fact=1
    for s in v1.shape:
        fact=fact*s
    return np.linalg.norm(v1.flatten()-v2.flatten())/fact

def hammingDistance(v1,v2):
    return hamming(v1.flatten(),v2.flatten())
