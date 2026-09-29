import numpy as np
from ppca import PPCA
from sklearn.decomposition import PCA
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

pca=PCA(n_components=17)

pc1=pca.fit_transform(img1)
pc2=pca.fit_transform(img2)
pc3=pca.fit_transform(img3)
pc4=pca.fit_transform(img4)
pc5=pca.fit_transform(img5)
'''
# 1. Simulating your dataset: 5 images (rows) x 17 PC features (columns)
np.random.seed(42)
pc_scores = np.random.randn(32,17)
pc_scores[pc_scores < -0.5] = np.nan  # Inject missing NaN values
'''
pc_scores=pc1

pc_scores[pc_scores < -0.5] = np.nan
# 2. Initialize the PPCA model
# Note: 'k' is the number of latent dimensions to fit. 
# Since you only have 5 images, 'k' must be strictly less than 5 (e.g., k=2 or k=3).
ppca_model = PPCA()

# 3. Fit the model on the data containing NaNs
# PPCA uses the EM algorithm to estimate the principal subspaces despite missing data
ppca_model.fit(data=pc_scores, d=2, verbose=True)
# 4. Reconstruct the complete dataset
# This fills in the NaNs using the learned probabilistic distribution
imputed_pc_scores = ppca_model.transform()

vals=np.where(pc_scores==np.nan,pc_scores)

pc_scores[vals]=imputed_pc_scores

recovered=pca.inverse_transform(pc_scores)

print("Original Matrix Shape:", pc_scores.shape)
print("Imputed Matrix Shape:", imputed_pc_scores.shape)
print("Any missing values left?", np.isnan(imputed_pc_scores).any())
