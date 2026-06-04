import joblib

columns = joblib.load("model/columns.pkl")

for i, col in enumerate(columns):
    print(i, col)