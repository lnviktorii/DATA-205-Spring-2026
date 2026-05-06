# Self-Sufficiency Navigator

**Capstone Project - DATA 205, Montgomery College | Spring 2026**

## What This Project Does
A Streamlit web application that helps Montgomery County, MD residents 
understand their financial situation. Users input their household profile 
and receive:
- A predicted annual income range
- Their self-sufficiency gap (difference between predicted income and 
  the living-wage threshold for their family type)
- Top 3 personalized recommendations to close that gap

## Data Sources
- **IPUMS ACS** - U.S. Census microdata filtered to Montgomery County, MD
- **Maryland Self-Sufficiency Standard (2023)** - 17,256 rows, 700+ family 
  types covering housing, food, childcare, transportation, healthcare costs
- **BLS CPI/ECI API** - inflation data by category to update 2023 costs 
  to 2026 actuals and project 2027 scenarios

## Methods
- XGBoost regression (survey-weighted, HHWT as sample weight)
- Stratified 5-fold cross-validation
- SHAP values for model interpretability and recommendation engine
- BLS CPI inflation adjustment per cost category

## Tech Stack
Python · XGBoost · scikit-learn · SHAP · pandas · Plotly · 
Streamlit · BLS API · IPUMS · joblib · GitHub

## Status
🔄 In progress — Spring 2026

## Author
Viktoriia Lyon | [LinkedIn]([https://www.linkedin.com/in/viktoriia-lyon](https://www.linkedin.com/in/viktoriia-lyon-7a2683106/)
