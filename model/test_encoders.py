import joblib

encoders = joblib.load("encoders.pkl")

print(encoders.keys())