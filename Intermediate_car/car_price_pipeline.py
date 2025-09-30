# car_price_pipeline.py
"""
End-to-end Car Price Prediction pipeline.
- Expects 'car.csv' in same folder with columns:
  Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
- Produces: trained model saved as 'car_price_model.pkl'
"""
import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint as sp_randint, uniform as sp_uniform

# --------- SETTINGS ----------
import os
DATA_PATH = os.path.join(os.path.dirname(__file__), "car.csv")
MODEL_OUT = "car_price_model.pkl"
RANDOM_STATE = 42
CURRENT_YEAR = 2025   # use 2025 as reference (update as needed)

# --------- 1. Load dataset ----------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Place your dataset as '{DATA_PATH}' in the working directory.")

df = pd.read_csv(DATA_PATH)
print("Initial shape:", df.shape)
print(df.head(6))

# --------- 2. Basic cleaning & normalization ----------
# Strip column names and replace common NA tokens
df.columns = [c.strip() for c in df.columns]
df = df.replace(['NA', 'NaN', 'nan', ''], np.nan)

# Ensure numeric columns are numeric
for c in ['Year','Selling_Price','Present_Price','Kms_Driven','Owner']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows without target
df = df.dropna(subset=['Selling_Price']).reset_index(drop=True)

# --------- 3. Feature engineering ----------
# Car age (use CURRENT_YEAR). If Year missing, impute later.
df['Car_Age'] = CURRENT_YEAR - df['Year']
df['Car_Age'] = df['Car_Age'].where(df['Car_Age'] >= 0, np.nan)

# Log-transform kms driven to reduce skew
df['Kms_Driven_log'] = np.log1p(df['Kms_Driven'].fillna(0))

# Extract brand from Car_Name (first token)
df['Brand'] = df['Car_Name'].astype(str).apply(lambda x: x.split()[0].lower())

# Quick checks
print("\nMissing values per column:\n", df.isna().sum())
print("\nTarget stats -> mean: {:.3f}, median: {:.3f}, std: {:.3f}".format(
    df['Selling_Price'].mean(), df['Selling_Price'].median(), df['Selling_Price'].std()))

# --------- 4. Choose features & target ----------
TARGET = 'Selling_Price'
FEATURES = [
    'Present_Price',
    'Kms_Driven_log',
    'Car_Age',
    'Owner',
    'Fuel_Type',
    'Seller_Type',
    'Transmission',
    'Brand'
]
# Keep only features that exist in df
FEATURES = [f for f in FEATURES if f in df.columns]
X = df[FEATURES].copy()
y = df[TARGET].copy()

# --------- 5. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# --------- 6. Preprocessing pipelines ----------
numeric_features = [c for c in ['Present_Price','Kms_Driven_log','Car_Age','Owner'] if c in X.columns]
categorical_features = [c for c in ['Fuel_Type','Seller_Type','Transmission','Brand'] if c in X.columns]

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
    ('cat', categorical_transformer, categorical_features)
])

# --------- 7. Baseline model (Linear Regression) ----------
baseline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('reg', LinearRegression())
])
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)
print("\nBaseline LinearRegression -> MAE: {:.3f}, RMSE: {:.3f}, R2: {:.3f}".format(
    mean_absolute_error(y_test, y_pred_baseline),
    np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
    r2_score(y_test, y_pred_baseline)
))

# --------- 8. Model selection: RF and GB (with small CV) ----------
models = {
    'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_STATE)
}

for name, m in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', m)])
    # 4-fold CV RMSE
    #cv_scores = cross_val_score(pipe, X_train, y_train, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1)
    #print(f"\n{name} CV RMSE: { -cv_scores.mean():.4f } ± { cv_scores.std():.4f }")
    import numpy as np

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=4, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_cv = np.sqrt(-cv_scores)  # take square root to convert MSE → RMSE
    print(f"\n{name} CV RMSE: {rmse_cv.mean():.4f} ± {rmse_cv.std():.4f}")


# --------- 9. Hyperparameter tuning for RandomForest (RandomizedSearchCV) ----------
rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('rf', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])

param_dist = {
    'rf__n_estimators': sp_randint(100, 500),
    'rf__max_depth': sp_randint(3, 25),
    'rf__min_samples_split': sp_randint(2, 10),
    'rf__min_samples_leaf': sp_randint(1, 6),
    'rf__max_features': ['auto', 'sqrt', 'log2', 0.6, 0.8]
}

rs = RandomizedSearchCV(
    rf_pipe,
    param_distributions=param_dist,
    n_iter=40,
    cv=4,
    scoring='neg_root_mean_squared_error',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)
print("\nStarting RandomizedSearchCV (RandomForest). This may take some minutes depending on data & machine...")
rs.fit(X_train, y_train)
print("Best RF params:", rs.best_params_)
print("Best RF CV RMSE:", -rs.best_score_)

# --------- 10. Evaluate best RF on test set ----------
best_rf = rs.best_estimator_
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
#rmse = mean_squared_error(y_test, y_pred, squared=False)
import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")

r2 = r2_score(y_test, y_pred)
print("\nBest RF Test -> MAE: {:.3f}, RMSE: {:.3f}, R2: {:.3f}".format(mae, rmse, r2))

# Cross-validated RMSE on train for stability
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
print("Train CV RMSE: {:.3f} ± {:.3f}".format(-cv_scores.mean(), cv_scores.std()))

# --------- 11. Feature importance (if tree-based) ----------
model_obj = best_rf.named_steps['rf']
if hasattr(model_obj, "feature_importances_"):
    # Build feature names after preprocessing: numeric + OHE cat names
    num_names = numeric_features
    ohe = best_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_ohe_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = num_names + cat_ohe_names
    importances = model_obj.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop feature importances (RF):")
    print(fi.head(15))
else:
    print("Model has no feature_importances_ attribute; consider permutation importance or SHAP.")

# --------- 12. Save best model pipeline ----------
joblib.dump(best_rf, MODEL_OUT)
print(f"\nSaved trained pipeline to '{MODEL_OUT}'")

# --------- 13. Helper: single prediction using saved pipeline ----------
def predict_single(car_dict, model_path=MODEL_OUT):
    """
    car_dict example:
    {
      "Car_Name": "ritz",
      "Year": 2014,
      "Present_Price": 5.59,
      "Kms_Driven": 27000,
      "Fuel_Type": "Petrol",
      "Seller_Type": "Dealer",
      "Transmission": "Manual",
      "Owner": 0
    }
    """
    m = joblib.load(model_path)
    df_in = pd.DataFrame([car_dict])
    # create engineered features exactly as training
    df_in['Kms_Driven_log'] = np.log1p(df_in['Kms_Driven'].fillna(0))
    df_in['Car_Age'] = CURRENT_YEAR - df_in['Year']
    df_in['Brand'] = df_in['Car_Name'].astype(str).apply(lambda x: x.split()[0].lower())
    features_in = [f for f in FEATURES if f in df_in.columns]
    X_in = df_in[features_in]
    pred = m.predict(X_in)[0]
    return float(pred)

# Example usage (uncomment to test)
example = {
    "Car_Name": "ritz",
    "Year": 2014,
    "Present_Price": 5.59,
    "Kms_Driven": 27000,
    "Fuel_Type": "Petrol",
    "Seller_Type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0
}
print("\nExample predicted Selling_Price:", predict_single(example))
