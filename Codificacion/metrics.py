import numpy as np
from scipy.spatial.distance import hamming
def euclideanDistance(v1,v2):
    return np.linalg.norm(v1.flatten()-v2.flatten())

def hammingDistance(v1,v2):
    return hamming(v1.flatten(),v2.flatten())