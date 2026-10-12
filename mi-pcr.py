import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def mi_pcr_impute(X_miss, n_components=2, max_iter=10, random_state=42):
    """
    Performs Variable-by-Variable Multiple Imputation via Principal Component Regression (MI-PCR-VBV)
    """
    np.random.seed(random_state)
    
    # CRITICAL FIX: Explicitly enforce a deep copy of the array to make it writeable
    if isinstance(X_miss, pd.DataFrame):
        X_filled = X_miss.to_numpy(copy=True)
    else:
        X_filled = np.array(X_miss, copy=True, dtype=float)
    
    # 1. Create original missing mask and handle initial baseline mean imputation
    missing_mask = np.isnan(X_filled)
    col_means = np.nanmean(X_filled, axis=0)
    
    # FIX: Flatten the tuple to get a simple array of column indices
    columns_with_missing = np.where(missing_mask.any(axis=0))[0]
    
    # FIX: Corrected range boundary to loop through total number of columns
    n_cols = X_filled.shape[1]
    for col_idx in range(n_cols):
        col_nan_mask = missing_mask[:, col_idx]
        if col_nan_mask.any():
            # If the entire column is missing, fall back to global mean zero
            fill_val = col_means[col_idx] if not np.isnan(col_means[col_idx]) else 0.0
            X_filled[col_nan_mask, col_idx] = fill_val
            
    # 2. Iterative MICE cycle
    for iteration in range(max_iter):
        for target_col in columns_with_missing:
            
            # Isolate all predictor columns (every column except the target)
            predictor_indices = [c for c in range(n_cols) if c != target_col]
            X_predictors = X_filled[:, predictor_indices]
            
            # Step A: Run PCA on clean predictor space
            pca = PCA(n_components=n_components, random_state=random_state)
            X_pcs = pca.fit_transform(X_predictors)
            
            # Step B: Segment based on original missing rows for this target column
            is_missing = missing_mask[:, target_col]
            
            # Train only on rows that were natively observed
            X_train = X_pcs[~is_missing]
            y_train = X_filled[~is_missing, target_col]
            
            # Target rows needing prediction
            X_predict = X_pcs[is_missing]
            
            if len(X_predict) == 0:
                continue
                
            # Step C: Fit the regression model
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            
            # Step D: Bayesian Stochastic Noise addition to protect data variance
            y_pred = lr.predict(X_predict)
            residuals = y_train - lr.predict(X_train)
            
            # Safeguard variance calculation against edge cases with few rows
            resid_std = np.std(residuals) if len(residuals) > 1 else 1e-4
            noise = np.random.normal(0, resid_std, size=len(y_pred))
            
            # Inject cleanly back into the master array
            X_filled[is_missing, target_col] = y_pred + noise
            
    # Return as original data structure type
    if isinstance(X_miss, pd.DataFrame):
        return pd.DataFrame(X_filled, columns=X_miss.columns, index=X_miss.index)
    return X_filled

# --- Execution Test Block ---
np.random.seed(42)
n_samples, n_features = 100, 10
latent = np.random.normal(0, 1, size=(n_samples, 3))
data = latent @ np.random.uniform(0.5, 2.0, size=(3, n_features)) + np.random.normal(0, 0.1, size=(n_samples, n_features))

df = pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(n_features)])
print(df.shape)
# Introduce 20% Random Missing Values
df_missing = df.copy()
mask = np.random.rand(*df_missing.shape) < 0.20
df_missing[mask] = np.nan

# Execute robust imputation code
df_imputed = mi_pcr_impute(df_missing, n_components=2, max_iter=5)

print("Imputation completed successfully!")
print(f"Remaining Missing Values: {df_imputed.isna().sum().sum()}")
print("\nSample Output (First 3 rows, 4 columns):\n", df_imputed.iloc[:3, :4])
