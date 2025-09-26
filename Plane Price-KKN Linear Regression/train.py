import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("Plane Price.csv")


# --- Data Cleaning ---
# Remove commas and extract numeric values safely
def clean_numeric(col):
    return pd.to_numeric(
        df[col].astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )

numeric_cols = [
    "HP or lbs thr ea engine", "Max speed Knots", "All eng rate of climb",
    "Landing over 50ft", "Empty weight lbs", "Length ft/in",
    "Wing span ft/in", "Range N.M."
]

for col in numeric_cols:
    df[col] = clean_numeric(col)

# Drop rows with missing target
df = df.dropna(subset=["Price"])

# Encode categorical column
le_engine = LabelEncoder()
df["Engine Type"] = le_engine.fit_transform(df["Engine Type"])

# Drop Model Name (too many categories)
df = df.drop(columns=["Model Name"])

# Features & Target
X = df.drop(columns=["Price"])
y = df["Price"]

# Handle missing values in features
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Find Best K Value ---
best_k = 1
best_score = -999

for k in range(1, 21):  # try k = 1 to 20
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="r2")
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_k = k

print(f"✅ Best K value: {best_k} with CV R2 score: {best_score:.4f}")

# --- Train Final Model ---
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# Evaluate on test data
y_pred = knn.predict(X_test_scaled)
print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R2 Score:", r2_score(y_test, y_pred))

import pickle

# Save trained model, scaler, imputer, and encoder
with open("model.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(le_engine, f)



'''
# --- Prediction Function ---
def predict_price(user_input):
    
    temp = user_input.copy()
    temp["Engine Type"] = le_engine.transform([temp["Engine Type"]])[0]

    # Convert to DataFrame -> Impute -> Scale
    user_df = pd.DataFrame([temp])
    user_df = imputer.transform(user_df)   # fill missing if any
    user_scaled = scaler.transform(user_df)

    return knn.predict(user_scaled)[0]

# Example usage
example_input = {
    'Engine Type': 'Propjet',
    'HP or lbs thr ea engine': 85,
    'Max speed Knots': 88,
    'Rcmnd cruise Knots': 78,
    'Stall Knots dirty': 37,
    'Fuel gal/lbs': 19,
    'All eng rate of climb': 620,
    'Eng out rate of climb': 500,
    'Takeoff over 50ft': 850,
    'Landing over 50ft': 1300,
    'Empty weight lbs': 800,
    'Length ft/in': 21.5,
    'Wing span ft/in': 35.0,
    'Range N.M.': 210
}
print("Predicted Price:", predict_price(example_input))
'''