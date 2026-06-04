import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv(
    "../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill Missing Values
df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)

# Encode Target
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Encode Categorical Features
label_encoders = {}

for col in df.select_dtypes(include="object").columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    label_encoders[col] = encoder
joblib.dump(
    label_encoders,
    "encoders.pkl"
)
# Features & Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save Feature Columns
joblib.dump(
    X.columns.tolist(),
    "columns.pkl"
)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy:.4f}")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Save Model
joblib.dump(
    model,
    "model.pkl"
)

print("Model Saved Successfully!")
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")
joblib.dump(label_encoders, "encoders.pkl")