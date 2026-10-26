import numpy as np
import pandas as pd
from ppca import PPCA
from nipals import nipals
from cv2 import imshow, waitKey, imread,imwrite
from Codificacion.process import normalizeIris

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

n_dim=4

nip=nipals.Nipals(pd.DataFrame(img1))
'''
extra=[]

for i in range(n_dim):
    extra.append(np.nan)

imputed=[]


for i in range(n_dim):
    X=[pc1[i],pc2[i],pc3[i],pc4[i],pc5[i],extra]

    nip=nipals.Nipals(X)
   
    ImputedData = nip.pred

    imputed.append(ImputedData[5])
    print(f'{i+1}/{n_dim}')
'''
nip.fit(ncomp=n_dim,cv=True)

nip.predict(pd.DataFrame(img2))
print(nip.pred)