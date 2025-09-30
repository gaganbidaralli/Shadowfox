# boston_regression_pipeline.py
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# Modeling & evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ---------- 1. Load data ----------
import os
fn = os.path.join(os.path.dirname(__file__), "boston.csv")

if not os.path.exists(fn):
    raise FileNotFoundError(f"Please place the dataset as '{fn}' in the working directory.")

# Read CSV; ensure proper parsing of 'NA' strings
df = pd.read_csv(fn, sep=None, engine='python')  # auto-detect separator
# Normalize column names
df.columns = [c.strip() for c in df.columns]
print("Data shape:", df.shape)
print(df.head())

# ---------- 2. Quick EDA & cleaning ----------
# Replace common 'NA' strings with real NaN
df = df.replace(['NA', 'NaN', 'nan', ''], np.nan)

# Show missing values
print("\nMissing values per column:\n", df.isna().sum())

# If MEDV is the target, ensure it's numeric
target_col = "MEDV"
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

# Drop rows with missing target
df = df.dropna(subset=[target_col]).reset_index(drop=True)

# Convert CHAS to categorical (0/1). Some rows show NA in sample -> impute later.
if 'CHAS' in df.columns:
    df['CHAS'] = pd.to_numeric(df['CHAS'], errors='coerce')

# Optionally inspect distributions: (commented out for script)
# df.hist(figsize=(12,10))

# ---------- 3. Handle outliers (basic) ----------
# We'll use IQR-based capping (winsorization) for numeric features to reduce extreme outliers' effect.
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove(target_col)
# Cap using 1st/99th percentiles or IQR method
for c in num_cols:
    # skip binary/categorical-like columns
    if df[c].nunique() <= 3:
        continue
    lower = df[c].quantile(0.01)
    upper = df[c].quantile(0.99)
    df[c] = df[c].clip(lower, upper)

# ---------- 4. Split X/y ----------
X = df.drop(columns=[target_col])
y = df[target_col]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 5. Preprocessing pipeline ----------
# Numeric and categorical feature lists
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
# CHAS might be numeric but is essentially categorical (0/1)
cat_features = [c for c in X.columns if c not in numeric_features]
# If CHAS exists but is numeric, treat it as categorical
if 'CHAS' in numeric_features:
    numeric_features.remove('CHAS')
    cat_features.append('CHAS')

print("Numeric features:", numeric_features)
print("Categorical features:", cat_features)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, cat_features)
])

# ---------- 6. Model candidates ----------
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

def evaluate_model(pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    mse = mean_squared_error(yte, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(yte, preds)
    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'y_pred': preds}

# ---------- 7. Cross-validate each model ----------
cv_results = {}
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    # Use negative MSE and R2 via cross_val_score (5-fold)
    scores_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    scores_mse = cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_results[name] = {
        'r2_mean': scores_r2.mean(),
        'r2_std': scores_r2.std(),
        'rmse_mean': -scores_mse.mean(),
        'rmse_std': scores_mse.std()
    }
    print(f"{name} CV R2: {scores_r2.mean():.4f} ± {scores_r2.std():.4f}, CV RMSE: {-scores_mse.mean():.4f}")

# ---------- 8. Pick best model and fine-tune ----------
# Suppose GradientBoosting or RandomForest is best; we'll tune RandomForest and GradientBoosting.
# I'll do a small grid search for RandomForest and for GradientBoosting.

# RandomForest param grid (small)
rf_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 6, 10],
    'model__min_samples_split': [2, 5]
}
rf_pipe = Pipeline([('preprocessor', preprocessor),
                    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])

rf_search = GridSearchCV(rf_pipe, rf_param_grid, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
rf_search.fit(X_train, y_train)
print("RandomForest best params:", rf_search.best_params_)
print("RandomForest best CV RMSE:", -rf_search.best_score_)

# GradientBoosting param grid (small)
gb_param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 4]
}
gb_pipe = Pipeline([('preprocessor', preprocessor),
                    ('model', GradientBoostingRegressor(random_state=42))])

gb_search = GridSearchCV(gb_pipe, gb_param_grid, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
gb_search.fit(X_train, y_train)
print("GB best params:", gb_search.best_params_)
print("GB best CV RMSE:", -gb_search.best_score_)

# Choose the best of rf_search and gb_search by CV RMSE
best_search = rf_search if -rf_search.best_score_ < -gb_search.best_score_ else gb_search
print("Selected best model:", best_search.best_estimator_)

# ---------- 9. Final evaluation on test set ----------
final_pipe = best_search.best_estimator_
final_pipe.fit(X_train, y_train)
y_pred = final_pipe.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred)
print(f"\nTest set results -> RMSE: {rmse_test:.4f}, MSE: {mse_test:.4f}, R2: {r2_test:.4f}")

# ---------- 10. Feature importance ----------
# If tree-based model is used:
model_obj = final_pipe.named_steps['model']
if hasattr(model_obj, 'feature_importances_'):
    # Get feature names after preprocessing
    num_names = numeric_features
    if cat_features:
        ohe = final_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_ohe_names = list(ohe.get_feature_names_out(cat_features))
    else:
        cat_ohe_names = []
    feature_names = num_names + cat_ohe_names
    importances = model_obj.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop feature importances:\n", fi.head(15))
else:
    # Permutation importance as fallback (works for any fitted model)
    r = permutation_importance(final_pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    # Need final feature names similar to above
    num_names = numeric_features
    if cat_features:
        ohe = final_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_ohe_names = list(ohe.get_feature_names_out(cat_features))
    else:
        cat_ohe_names = []
    feature_names = num_names + cat_ohe_names
    perm_sorted_idx = r.importances_mean.argsort()[::-1]
    print("\nPermutation importances (mean) top features:")
    for idx in perm_sorted_idx[:15]:
        print(f"{feature_names[idx]}: mean={r.importances_mean[idx]:.4f} std={r.importances_std[idx]:.4f}")

# ---------- 11. Save model if desired ----------
import joblib
joblib.dump(final_pipe, "boston_price_model.pkl")
print("\nSaved final pipeline to 'boston_price_model.pkl'")
