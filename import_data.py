import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# PostgreSQL Password
password = quote_plus("Sr@22102003")

DATABASE_URL = (
    f"postgresql://postgres:{password}"
    "@localhost:5432/customer_churn_db"
)

engine = create_engine(DATABASE_URL)

# Load CSV
df = pd.read_csv(r"C:\Users\DELL\Desktop\Customer-Churn-Webapp\dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Upload to PostgreSQL
df.to_sql(
    "customers",
    engine,
    if_exists="replace",
    index=False
)

print("Data Imported Successfully!")
print(df.shape)