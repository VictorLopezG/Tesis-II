import numpy as np # linear algebra
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from cv2 import imshow, waitKey, imread,imwrite, createCLAHE
from scipy.linalg import svd
from Codificacion.process import normalizeIris
from Segmentacion.normalize import rubber_sheet_inverse

def SVDGen(paths:list[str]=None,imgs:list=None):
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

    img1,_=normalizeIris(img1)
    img2,_=normalizeIris(img2)
    img3,_=normalizeIris(img3)
    img4,_=normalizeIris(img4)
    img5,_=normalizeIris(img5)

    U1, S1, V1 = svd(img1,full_matrices=False)
    U2, S2, V2 = svd(img2,full_matrices=False)
    U3, S3, V3 = svd(img3,full_matrices=False)
    U4, S4, V4 = svd(img4,full_matrices=False)
    U5, S5, V5 = svd(img5,full_matrices=False)

    UI=np.full(U1.shape,np.nan)
    SI=np.full(S1.shape,np.nan)
    VI=np.full(V1.shape,np.nan)

    nU=[]
    nS=[]
    nV=[]

    X=[S1,S2,S3,S4,S5,SI]

    lr=LinearRegression()

    imp=IterativeImputer(estimator=lr, max_iter=5, tol=1e-10, imputation_order='random')

    ImputedData=imp.fit_transform(X)

    nS=ImputedData[5]

    for i in range(U1.shape[0]):
        X=[U1[i],U2[i],U3[i],U4[i],U5[i],UI[i]]

        rf = LinearRegression()
    
        imp = IterativeImputer(estimator=rf, max_iter=5, tol=1e-10, imputation_order='random')

        ImputedData = imp.fit_transform(X)

        nU.append(ImputedData[5])
        #print(f'{i+1}/{U1.shape[0]}')

    for i in range(V1.shape[0]):
        X=[V1[i],V2[i],V3[i],V4[i],V5[i],VI[i]]

        rf = LinearRegression()
    
        imp = IterativeImputer(estimator=rf, max_iter=5, tol=1e-10, imputation_order='random')

        ImputedData = imp.fit_transform(X)

        nV.append(ImputedData[5])

    synthetic_img = np.dot(nU, np.dot(np.diag(nS), nV))

    clahe=createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    synthetic_img=clahe.apply((synthetic_img*255).astype(np.uint8))

    return synthetic_img


"""
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

U1, S1, V1 = svd(img1,full_matrices=False)
U2, S2, V2 = svd(img2,full_matrices=False)
U3, S3, V3 = svd(img3,full_matrices=False)
U4, S4, V4 = svd(img4,full_matrices=False)
U5, S5, V5 = svd(img5,full_matrices=False)

#a=U1 @ (np.diag(S1)@V1)

#imshow("a",a)
#waitKey()

print(U1.shape,S1.shape,V1.shape)

UI=np.full(U1.shape,np.nan)
SI=np.full(S1.shape,np.nan)
VI=np.full(V1.shape,np.nan)

nU=[]
nS=[]
nV=[]

X=[S1,S2,S3,S4,S5,SI]

lr=LinearRegression()

imp=IterativeImputer(estimator=lr, max_iter=5, tol=1e-10, imputation_order='random')

ImputedData=imp.fit_transform(X)

nS=ImputedData[5]

for i in range(U1.shape[0]):
    X=[U1[i],U2[i],U3[i],U4[i],U5[i],UI[i]]

    rf = LinearRegression()
    
    imp = IterativeImputer(estimator=rf, max_iter=5, tol=1e-10, imputation_order='random')

    ImputedData = imp.fit_transform(X)

    nU.append(ImputedData[5])
    print(f'{i+1}/{U1.shape[0]}')

for i in range(V1.shape[0]):
    X=[V1[i],V2[i],V3[i],V4[i],V5[i],VI[i]]

    rf = LinearRegression()
    
    imp = IterativeImputer(estimator=rf, max_iter=5, tol=1e-10, imputation_order='random')

    ImputedData = imp.fit_transform(X)

    nV.append(ImputedData[5])
    print(f'{i+1}/{V1.shape[0]}')

print(np.asarray(nU).shape)
print(np.asarray(nS).shape)
print(np.asarray(nV).shape)

synthetic_img = np.dot(nU, np.dot(np.diag(nS), nV))

clahe=createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

synthetic_img=clahe.apply((synthetic_img*255).astype(np.uint8))

imwrite("imputedImgs/SVDII.jpg",synthetic_img)

circ=rubber_sheet_inverse(synthetic_img,(290,320),50,200)

imwrite("imputedImgs/SVDIICirc.jpg",circ)
"""