# show_encoders.py

import joblib

encoders = joblib.load("encoders.pkl")

for column, encoder in encoders.items():
    print(f"\n{column}")
    print(list(encoder.classes_))