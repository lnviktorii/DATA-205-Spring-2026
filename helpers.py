"""Helper functions for the Self-Sufficiency Navigator app."""
import joblib
import json
import pandas as pd

# CPI adjustment 2023 -> 2026 (~8.3% cumulative)
CPI_2023_TO_2026 = 1.083


def load_model_and_features():
    """Load the trained XGBoost model and feature column list."""
    model = joblib.load("model.pkl")
    with open("feature_columns.json") as f:
        features = json.load(f)
    return model, features


def load_sss_table():
    """Load and clean the Self-Sufficiency Standard table."""
    df = pd.read_excel("MD2023_SSS.xlsx", sheet_name="By Family")
    df.columns = df.columns.str.strip()
    return df[df["County"] == "Montgomery County"].copy()


def get_sss_annual_adjusted(df_mc, adults, infants, preschoolers, 
                             schoolagers, teenagers):
    """SSS annual wage adjusted from 2023 to 2026. Returns None if no match."""
    match = df_mc[
        (df_mc["Adult(s)"] == adults)
        & (df_mc["Infant(s)"] == infants)
        & (df_mc["Preschooler(s)"] == preschoolers)
        & (df_mc["Schoolager(s)"] == schoolagers)
        & (df_mc["Teenager(s)"] == teenagers)
    ]
    if len(match) == 0:
        return None
    return float(match["Annual Self-Sufficiency Wage"].iloc[0]) * CPI_2023_TO_2026


def predict_income(model, features, user_inputs):
    """Predict annual earned income from user inputs."""
    row = pd.DataFrame([user_inputs])[features]
    return float(model.predict(row)[0])