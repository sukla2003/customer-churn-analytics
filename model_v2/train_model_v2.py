import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv(
    "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Drop ID
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill Missing
df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)

# Target
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop("Churn", axis=1)

# Feature Types
categorical_features = X.select_dtypes(
    include="object"
).columns

numerical_features = X.select_dtypes(
    exclude="object"
).columns

# Preprocessing
preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",
            Pipeline([
                ("imputer",
                 SimpleImputer(
                     strategy="most_frequent"
                 )),
                ("encoder",
                 OneHotEncoder(
                     handle_unknown="ignore"
                 ))
            ]),
            categorical_features
        ),

        (
            "num",
            Pipeline([
                ("imputer",
                 SimpleImputer(
                     strategy="median"
                 ))
            ]),
            numerical_features
        )

    ]

)

pipeline = Pipeline([

    ("preprocessor", preprocessor),

    ("classifier",
     RandomForestClassifier(
         random_state=42
     ))

])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# Hyperparameter Tuning
params = {

    "classifier__n_estimators":
        [100, 200],

    "classifier__max_depth":
        [10, 20, None],

    "classifier__min_samples_split":
        [2, 5]

}

grid = GridSearchCV(

    pipeline,
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1

)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

preds = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    preds
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    "Best Parameters:",
    grid.best_params_
)

joblib.dump(
    best_model,
    "churn_pipeline.pkl"
)

print(
    "Improved Model Saved!"
)