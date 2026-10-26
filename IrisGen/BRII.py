import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Explicitly enable MICE
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from Codificacion.process import normalizeIris
from cv2 import imread, imwrite
from sklearn.decomposition import PCA

n_dim=64

img1 = imread("C:\\Users\\wings\\Documents\\Codigos\\Desarrrollo\\sistema_reconocimiento_iris\\CASIA\\TEST\\10\\010_1_3.jpg",0)
img2 = imread("C:\\Users\\wings\\Documents\\Codigos\\Desarrrollo\\sistema_reconocimiento_iris\\CASIA\\TEST\\10\\010_2_3.jpg",0)
img3 = imread("C:\\Users\\wings\\Documents\\Codigos\\Desarrrollo\\sistema_reconocimiento_iris\\CASIA\\TEST\\10\\010_2_4.jpg",0)
img4 = imread("C:\\Users\\wings\\Documents\\Codigos\\Desarrrollo\\sistema_reconocimiento_iris\\CASIA\\TEST\\11\\011_1_3.jpg",0)
img5 = imread("C:\\Users\\wings\\Documents\\Codigos\\Desarrrollo\\sistema_reconocimiento_iris\\CASIA\\TEST\\11\\011_2_3.jpg",0)

img1,_=normalizeIris(img1)
img2,_=normalizeIris(img2)
img3,_=normalizeIris(img3)
img4,_=normalizeIris(img4)
img5,_=normalizeIris(img5)

pca=PCA(n_components=n_dim,whiten=True)

pc1=pca.fit_transform(img1)
pc2=pca.fit_transform(img2)
pc3=pca.fit_transform(img3)
pc4=pca.fit_transform(img4)
pc5=pca.fit_transform(img5)

extra=[]

for i in range(n_dim):
    extra.append(np.nan)

mice_imputer = IterativeImputer(estimator=BayesianRidge(),max_iter=10,random_state=42)

imputed=[]

for i in range(n_dim):
    X=[pc1[i],pc2[i],pc3[i],pc4[i],pc5[i],extra]

    ImputedData=mice_imputer.fit_transform(X)

    imputed.append(ImputedData[5])
    print(f'{i+1}/{n_dim}')

regr=pca.inverse_transform(imputed)

imwrite("imputedImgs/BRII.jpg",regr*255)
