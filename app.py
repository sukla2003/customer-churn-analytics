from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib

from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os

app = Flask(__name__)

# ==================================
# LOAD PIPELINE MODEL
# ==================================

pipeline = joblib.load(
    "model_v2/churn_pipeline.pkl"
)

# ==================================
# DATABASE CONNECTION
# ==================================

password = quote_plus(
    os.environ.get("Sr@22102003")
)

DATABASE_URL = (
    f"postgresql://postgres:{password}"
    "@localhost:5432/customer_churn_db"
)

engine = create_engine(DATABASE_URL)

# ==================================
# HOME PAGE
# ==================================

@app.route("/")
def home():
    return render_template("index.html")

# ==================================
# PREDICTION ROUTE
# ==================================

@app.route("/predict", methods=["POST"])
def predict():

    input_df = pd.DataFrame([{

        "gender":
            request.form["gender"],

        "SeniorCitizen":
            0,

        "Partner":
            "No",

        "Dependents":
            "No",

        "tenure":
            int(request.form["tenure"]),

        "PhoneService":
            "Yes",

        "MultipleLines":
            "No",

        "InternetService":
            request.form["InternetService"],

        "OnlineSecurity":
            "No",

        "OnlineBackup":
            "No",

        "DeviceProtection":
            "No",

        "TechSupport":
            "No",

        "StreamingTV":
            "No",

        "StreamingMovies":
            "No",

        "Contract":
            request.form["Contract"],

        "PaperlessBilling":
            "Yes",

        "PaymentMethod":
            request.form["PaymentMethod"],

        "MonthlyCharges":
            float(request.form["MonthlyCharges"]),

        "TotalCharges":
            float(request.form["TotalCharges"])

    }])

    prediction = pipeline.predict(
        input_df
    )[0]

    probability = pipeline.predict_proba(
        input_df
    )[0][1]

    probability = float(probability)

    if prediction == 1:
        result = "⚠ Customer Likely To Churn"
        db_prediction = "Churn"
    else:
        result = "✅ Customer Likely To Stay"
        db_prediction = "Stay"

    with engine.connect() as conn:

        conn.execute(
            text("""
                INSERT INTO predictions
                (prediction, probability)
                VALUES
                (:prediction, :probability)
            """),
            {
                "prediction": db_prediction,
                "probability": round(probability * 100, 2)
            }
        )

        conn.commit()

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 2)
    )

# ==================================
# HISTORY PAGE
# ==================================

@app.route("/history")
def history():

    search = request.args.get(
        "search",
        ""
    )

    with engine.connect() as conn:

        if search:

            rows = conn.execute(
                text("""
                    SELECT *
                    FROM predictions
                    WHERE prediction ILIKE :search
                    ORDER BY id DESC
                """),
                {
                    "search":
                        f"%{search}%"
                }
            ).fetchall()

        else:

            rows = conn.execute(
                text("""
                    SELECT *
                    FROM predictions
                    ORDER BY id DESC
                """)
            ).fetchall()

    return render_template(
        "history.html",
        predictions=rows
    )

# ==================================
# DASHBOARD PAGE
# ==================================

@app.route("/dashboard")
def dashboard():

    with engine.connect() as conn:

        total = conn.execute(
            text(
                "SELECT COUNT(*) FROM predictions"
            )
        ).scalar()

        churn = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM predictions
                WHERE prediction='Churn'
            """)
        ).scalar()

        stay = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM predictions
                WHERE prediction='Stay'
            """)
        ).scalar()

        avg_probability = conn.execute(
            text("""
                SELECT COALESCE(
                    ROUND(AVG(probability),2),
                    0
                )
                FROM predictions
            """)
        ).scalar()

        high_risk = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM predictions
                WHERE probability >= 80
            """)
        ).scalar()

        recent_predictions = conn.execute(
            text("""
                SELECT *
                FROM predictions
                ORDER BY id DESC
                LIMIT 5
            """)
        ).fetchall()

        trend_data = conn.execute(
            text("""
                SELECT
                    DATE(created_at) AS day,
                    COUNT(*) AS total
                FROM predictions
                GROUP BY DATE(created_at)
                ORDER BY day
            """)
        ).fetchall()

    trend_labels = [
        str(row.day)
        for row in trend_data
    ]

    trend_values = [
        row.total
        for row in trend_data
    ]

    return render_template(
        "dashboard.html",
        total=total,
        churn=churn,
        stay=stay,
        avg_probability=avg_probability,
        high_risk=high_risk,
        recent_predictions=recent_predictions,
        trend_labels=trend_labels,
        trend_values=trend_values
    )

# ==================================
# CSV EXPORT
# ==================================

@app.route("/export")
def export_csv():

    query = """
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """

    df = pd.read_sql(
        query,
        engine
    )

    filename = "predictions.csv"

    df.to_csv(
        filename,
        index=False
    )

    return send_file(
        filename,
        as_attachment=True
    )

# ==================================
# INSIGHTS PAGE
# ==================================

@app.route("/insights")
def insights():

    with engine.connect() as conn:

        total = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM predictions
            """)
        ).scalar()

        high_risk = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM predictions
                WHERE probability >= 80
            """)
        ).scalar()

        avg_probability = conn.execute(
            text("""
                SELECT COALESCE(
                    ROUND(AVG(probability),2),
                    0
                )
                FROM predictions
            """)
        ).scalar()

        churn_predictions = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM predictions
                WHERE prediction='Churn'
            """)
        ).scalar()

    churn_rate = 0

    if total > 0:
        churn_rate = round(
            (churn_predictions / total) * 100,
            2
        )

    # Simulated business metric
    revenue_at_risk = round(
        high_risk * 75,
        2
    )

    return render_template(
        "insights.html",
        total=total,
        high_risk=high_risk,
        avg_probability=avg_probability,
        churn_rate=churn_rate,
        revenue_at_risk=revenue_at_risk
    )

# ==================================
# RUN APP
# ==================================

if __name__ == "__main__":
    app.run(debug=True)