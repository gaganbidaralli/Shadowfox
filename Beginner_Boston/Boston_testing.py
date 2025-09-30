import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("boston_price_model.pkl")

# Example new data (must match feature columns)
new_house = pd.DataFrame([{
    'CRIM': 0.05,
    'ZN': 12.5,
    'INDUS': 7.87,
    'CHAS': 0,
    'NOX': 0.524,
    'RM': 6.2,
    'AGE': 65,
    'DIS': 5.0,
    'RAD': 5,
    'TAX': 311,
    'PTRATIO': 15.2,
    'B': 390.5,
    'LSTAT': 12.0
}])

# Predict price
pred_price = model.predict(new_house)
print("Predicted House Price:", pred_price[0])
