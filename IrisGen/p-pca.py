import numpy as np
from ppca import PPCA
from sklearn.decomposition import PCA
from cv2 import imshow, waitKey, imread,imwrite
from Codificacion.process import normalizeIris
from random import randrange

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

pca=PCA(n_components=2)

pc1=pca.fit_transform(img1)
pc2=pca.fit_transform(img2)
pc3=pca.fit_transform(img3)
pc4=pca.fit_transform(img4)
pc5=pca.fit_transform(img5)

#print(pc1)

imps=[] 
#extra=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
'''
for i in range(17):
    new= np.append([pc1[i],pc2[i],pc3[i],pc4[i]],[pc5[i]],axis=0)
    imps.append(new)
'''
new=np.append([pc1[randrange(17)],pc2[randrange(17)],pc3[randrange(17)],pc4[randrange(17)]],[pc5[randrange(17)]],axis=0)
#print(imps[0])

ppca_model = PPCA()

# 3. Fit the model on the data containing NaNs
# PPCA uses the EM algorithm to estimate the principal subspaces despite missing data
'''
for i in range(17):
    ppca_model.fit(data=imps[i+1], d=2, verbose=True)
# 4. Reconstruct the complete dataset
# This fills in the NaNs using the learned probabilistic distribution
    imputed_pc_scores = ppca_model.transform()

    print("Original Matrix Shape:", pc_scores.shape)
    print("Imputed Matrix Shape:", imputed_pc_scores.shape)
'''
ppca_model.fit(data=new, d=2, verbose=True)
imputed_pc_scores = ppca_model.transform()
print(imputed_pc_scores)