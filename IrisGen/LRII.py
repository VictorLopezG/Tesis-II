import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from cv2 import imshow, waitKey, imread,imwrite
from sklearn.decomposition import PCA
from Codificacion.process import normalizeIris
from Segmentacion.normalize import rubber_sheet_inverse

def PCAGen(paths:list[str]=None,imgs:list=None):
    if len(imgs)>=5 and paths==None:
        img1=imgs[0]
        img2=imgs[1]
        img3=imgs[2]
        img4=imgs[3]
        img5=imgs[4]
    elif len(paths)>=5:
        img1 = imread(paths[0],0)
        img2 = imread(paths[1],0)
        img3 = imread(paths[2],0)
        img4 = imread(paths[3],0)
        img5 = imread(paths[4],0)
    else:
        return 0
    
    n_dim=64

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

    imputed=[]


    for i in range(n_dim):
        X=[pc1[i],pc2[i],pc3[i],pc4[i],pc5[i],extra]

        rf = LinearRegression()
    
        imp = IterativeImputer(estimator=rf, max_iter=5, tol=1e-10, imputation_order='random')

        ImputedData = imp.fit_transform(X)

        imputed.append(ImputedData[5])

    regr=pca.inverse_transform(imputed)

    return regr*255

"""
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

imputed=[]


for i in range(n_dim):
    X=[pc1[i],pc2[i],pc3[i],pc4[i],pc5[i],extra]

    rf = LinearRegression()
    
    imp = IterativeImputer(estimator=rf, max_iter=5, tol=1e-10, imputation_order='random')

    ImputedData = imp.fit_transform(X)

    imputed.append(ImputedData[5])
    print(f'{i+1}/{n_dim}')

regr=pca.inverse_transform(imputed)

print(img1.shape)

print(regr.shape)

imwrite("imputedImgs/LRII.jpg",regr*255)

circ=rubber_sheet_inverse(regr*255,(290,320),50,200)

imwrite("imputedImgs/LRIICirc.jpg",circ)
"""
