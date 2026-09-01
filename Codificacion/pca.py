from sklearn.decomposition import PCA
import numpy as np

def prCoAn(polar_array):
    pca=PCA(n_components=32)
    temp=pca.fit_transform(polar_array)
    #mask=pca.inverse_transform(temp)
    mask=np.ones(temp.shape,np.int8)
    return temp,mask