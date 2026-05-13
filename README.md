# Self-Sufficiency Navigator

Capstone Project, DATA 205, Montgomery College, Spring 2026

## What This Project Does

A Streamlit web application that helps Montgomery County, MD residents understand their financial situation. Users enter their household profile and receive:

- A predicted annual earned income from an XGBoost regression model
- The Self-Sufficiency Standard (cost of living) for their household composition
- The gap between predicted income and self-sufficiency need
- Actionable recommendations that close the gap

## Data Sources

- **IPUMS USA 2023 ACS microdata** filtered to Maryland working-age adults (18 to 64), 28,077 rows
- **Maryland Self-Sufficiency Standard (2023)** by Dr. Diana Pearce, University of Washington Center for Women's Welfare, 719 family-type rows for Montgomery County
- **BLS CPI and Employment Cost Index** for 2023 to 2026 temporal adjustment

## Methods

- XGBoost regression on log-transformed income (log1p of INCTOT)
- Training data filtered to incomes at or below $250K (covers all SSS thresholds for Montgomery County)
- 80/20 train/test split, 20 features including immigration variables (FOREIGN_BORN, CITIZEN, SPEAKENG, YRSUSA1)
- Feature importance computed from XGBoost's built-in importances
- Temporal adjustment uses CPI for cost-of-living and ECI for wage growth (different indices because wages and prices grow at different rates)

## Model Performance

| Metric | Value |
|--------|-------|
| Test MAE | $19,622 |
| Test RMSE | $32,673 |
| R² (dollar-space) | 0.615 |
| R² (log-space) | 0.804 |

## Tech Stack

Python, XGBoost, scikit-learn, pandas, Streamlit, matplotlib, joblib, GitHub

## Status

Complete, Spring 2026

## Author

Viktoriia Lyon, [LinkedIn](https://www.linkedin.com/in/viktoriia-lyon-7a2683106/)