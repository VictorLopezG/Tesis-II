import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from cv2 import imshow, waitKey, imread,imwrite
from sklearn.decomposition import PCA
from Codificacion.process import normalizeIris
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
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

X=[pc1,pc2,pc3,pc4,pc5]

rf = RandomForestRegressor(n_estimators=100)
imp = IterativeImputer(estimator=rf, verbose=2, max_iter=5, tol=1e-10, imputation_order='roman')

ImputedData = imp.fit_transform(X)

Imputed_data = pd.DataFrame(ImputedData)
print(Imputed_data)

'''
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Explicitly enable MICE
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# 1. Assume 'pc_scores' is your 2D NumPy array or Pandas DataFrame of extracted image PCs
# Context: Shape (n_images, n_components) with some missing NaN values
pc_scores = np.array([
    [1.5, -0.4, np.nan],
    [-2.1, np.nan, 0.1],
    [np.nan, 1.2, -0.5],
    [0.9, -0.8, 0.4]
])

# 2. Configure the MICE algorithm
# Using BayesianRidge as the estimator for continuous, orthogonal PC scores
mice_imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42
)

# 3. Fit and transform the data to fill missing values
imputed_pc_scores = mice_imputer.fit_transform(pc_scores)

print("Imputed PC Scores:\n", imputed_pc_scores)

'''

'''
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge

# Simulating your 5 images x 17 PC columns dataset with random NaNs
np.random.seed(42)
pc_scores = np.random.randn(5, 17)
pc_scores[pc_scores < -0.5] = np.nan  # Introduce missing values

# Configure MICE for high-dimensional, small-sample data
mice_imputer = IterativeImputer(
    # 1. Use Ridge regression with strong regularization to handle p > n
    estimator=Ridge(alpha=10.0), 
    
    # 2. Limit the number of predictors used for imputing any single column.
    # Since you only have 5 rows, each regression should use at most 2-3 predictors.
    n_nearest_features=2, 
    
    max_iter=15,
    random_state=42
)

# Fit and transform
imputed_pc_scores = mice_imputer.fit_transform(pc_scores)
print("Imputed Shape:", imputed_pc_scores.shape)

'''


'''
import numpy as np
from ppca import PPCA

# 1. Simulating your dataset: 5 images (rows) x 17 PC features (columns)
np.random.seed(42)
pc_scores = np.random.randn(5, 17)
pc_scores[pc_scores < -0.5] = np.nan  # Inject missing NaN values

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

print("Original Matrix Shape:", pc_scores.shape)
print("Imputed Matrix Shape:", imputed_pc_scores.shape)
print("Any missing values left?", np.isnan(imputed_pc_scores).any())

'''