import joblib
import pandas as pd

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

importance = pd.DataFrame({
    "Feature": columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(10))