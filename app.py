"""Self-Sufficiency Navigator — Streamlit App"""
import streamlit as st
import pandas as pd
from helpers import (
    load_model_and_features,
    load_sss_table,
    get_sss_annual_adjusted,
    predict_income,
)

st.set_page_config(page_title="Self-Sufficiency Navigator", 
                   page_icon="🧭", layout="wide")

# Load artifacts once and cache
@st.cache_resource
def load_all():
    model, features = load_model_and_features()
    df_mc = load_sss_table()
    return model, features, df_mc

model, features, df_mc = load_all()

# ============================================================
# HEADER
# ============================================================
st.title("🧭 Self-Sufficiency Navigator")
st.markdown(
    "**Montgomery County, MD** — Compare your predicted earned income "
    "against the cost of self-sufficiency for your household."
)
st.caption("Capstone Project • DATA 205 • Spring 2026 • Viktoriia Lyon")

st.divider()

# ============================================================
# PAGE 1: YOUR SITUATION (inputs)
# ============================================================
st.header("1. Your Situation")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("About you")
    age = st.slider("Age", 18, 64, 35)
    sex = st.selectbox("Sex", [(1, "Male"), (2, "Female")], 
                       format_func=lambda x: x[1])[0]
    educ_options = [
        (2, "Less than high school"),
        (6, "High school graduate"),
        (7, "Some college"),
        (10, "Associate's degree"),
        (11, "Bachelor's degree"),
        (14, "Master's degree"),
        (16, "Doctorate / Professional"),
    ]
    educ = st.selectbox("Education", educ_options, index=4,
                        format_func=lambda x: x[1])[0]

with col2:
    st.subheader("Your work")
    uhrswork = st.slider("Hours worked per week", 0, 80, 40)
    wkswork2_options = [
        (6, "50–52 weeks"),
        (5, "48–49 weeks"),
        (4, "40–47 weeks"),
        (3, "27–39 weeks"),
        (2, "14–26 weeks"),
        (1, "1–13 weeks"),
    ]
    wkswork2 = st.selectbox("Weeks worked per year", wkswork2_options,
                            format_func=lambda x: x[1])[0]
    classwkr = st.selectbox(
        "Worker type",
        [(2, "Employee (wage/salary)"), (1, "Self-employed")],
        format_func=lambda x: x[1])[0]

with col3:
    st.subheader("Your household")
    adults = st.number_input("Adults", 1, 5, 1)
    infants = st.number_input("Infants (0–2 years)", 0, 5, 0)
    preschoolers = st.number_input("Preschoolers (3–5)", 0, 5, 0)
    schoolagers = st.number_input("School-age (6–12)", 0, 5, 0)
    teenagers = st.number_input("Teenagers (13–17)", 0, 5, 0)

st.divider()

# ============================================================
# Build feature row for the model
# ============================================================
# Use median/mode values for features the user doesn't directly set
user_inputs = {
    "AGE": age,
    "SEX": sex,
    "RACE": 1,         # White (most common); could be added to UI
    "HISPAN": 0,       # Not Hispanic
    "MARST": 1 if adults >= 2 else 6,  # Married if 2+ adults, else single
    "FAMSIZE": adults + infants + preschoolers + schoolagers + teenagers,
    "NCHILD": infants + preschoolers + schoolagers + teenagers,
    "NCHLT5": infants + preschoolers,
    "EDUC": educ,
    "DEGFIELD": 0,     # Most common (no specific field)
    "OCC": 4700,       # Sales/office occupations (common)
    "IND": 8680,       # Services
    "CLASSWKR": classwkr,
    "UHRSWORK": uhrswork,
    "WKSWORK2": wkswork2,
    "PUMA": 1101,      # A Montgomery County PUMA
}

# ============================================================
# PAGE 2: PREDICTION
# ============================================================
st.header("2. Your Predicted Earned Income")

predicted_income = predict_income(model, features, user_inputs)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Predicted annual income", f"${predicted_income:,.0f}")
with m2:
    st.metric("Per month", f"${predicted_income/12:,.0f}")
with m3:
    st.metric("Hourly equivalent", 
              f"${predicted_income/(uhrswork*52) if uhrswork>0 else 0:,.2f}")

st.caption("Based on an XGBoost model trained on 28,077 working-age "
           "Marylanders from IPUMS USA 2023 ACS data. R² = 0.49.")

st.divider()

# ============================================================
# PAGE 3: THE GAP
# ============================================================
st.header("3. The Self-Sufficiency Gap")

sss_annual = get_sss_annual_adjusted(
    df_mc, adults, infants, preschoolers, schoolagers, teenagers
)

if sss_annual is None:
    st.warning(
        "Your household composition isn't covered by the Self-Sufficiency "
        "Standard data. Try a different combination of household members."
    )
else:
    gap = predicted_income - sss_annual
    
    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Predicted income", f"${predicted_income:,.0f}")
    with g2:
        st.metric("Self-sufficiency need (2026)", f"${sss_annual:,.0f}")
    with g3:
        st.metric("Gap", f"${gap:,.0f}", 
                  delta=f"${gap:,.0f}",
                  delta_color="normal" if gap >= 0 else "inverse")
    
    if gap >= 0:
        st.success(
            f"✅ Your predicted income exceeds the self-sufficiency "
            f"threshold by **${gap:,.0f}/year**."
        )
    else:
        st.error(
            f"⚠️ Your predicted income falls **${-gap:,.0f}/year** short "
            f"of the self-sufficiency threshold for your household."
        )
    
    # Visual comparison
    chart_df = pd.DataFrame({
        "Category": ["Predicted income", "Self-sufficiency need"],
        "Annual ($)": [predicted_income, sss_annual]
    })
    st.bar_chart(chart_df, x="Category", y="Annual ($)", horizontal=True)
    
    st.caption("Self-Sufficiency Standard from Dr. Diana Pearce, "
               "University of Washington Center for Women's Welfare, 2023, "
               "adjusted to 2026 using CPI-U for the Washington DMV area.")

st.divider()

# ============================================================
# PAGE 4: RECOMMENDATIONS
# ============================================================
st.header("4. Closing the Gap")

if sss_annual is None:
    st.info("Recommendations will appear once we have a valid household.")
elif gap >= 0:
    st.info("You're already at or above the self-sufficiency threshold. "
            "Consider strategies for savings, retirement, and emergency funds.")
else:
    st.write("**What if you changed one thing?** These scenarios re-run the "
             "model with one input changed to show potential impact:")
    
    scenarios = []
    
    # Scenario 1: full-time hours
    if uhrswork < 40:
        new_inputs = user_inputs.copy()
        new_inputs["UHRSWORK"] = 40
        new_pred = predict_income(model, features, new_inputs)
        scenarios.append(("Work full-time (40 hrs/week)", new_pred, 
                          new_pred - predicted_income))
    
    # Scenario 2: bachelor's degree
    if educ < 11:
        new_inputs = user_inputs.copy()
        new_inputs["EDUC"] = 11
        new_pred = predict_income(model, features, new_inputs)
        scenarios.append(("Earn a Bachelor's degree", new_pred,
                          new_pred - predicted_income))
    
    # Scenario 3: master's degree
    if educ < 14:
        new_inputs = user_inputs.copy()
        new_inputs["EDUC"] = 14
        new_pred = predict_income(model, features, new_inputs)
        scenarios.append(("Earn a Master's degree", new_pred,
                          new_pred - predicted_income))
    
    # Scenario 4: 50-52 weeks worked
    if wkswork2 < 6:
        new_inputs = user_inputs.copy()
        new_inputs["WKSWORK2"] = 6
        new_pred = predict_income(model, features, new_inputs)
        scenarios.append(("Work all 52 weeks", new_pred,
                          new_pred - predicted_income))
    
    if scenarios:
        for label, new_income, delta in scenarios:
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.write(f"**{label}**")
            with c2:
                st.write(f"New income: ${new_income:,.0f}")
            with c3:
                color = "🟢" if delta > 0 else "🔴"
                st.write(f"{color} {'+' if delta>=0 else ''}${delta:,.0f}/year")
    else:
        st.write("You're already at the highest values for these factors.")
    
    st.caption("These are estimates from the model — individual results vary "
               "based on field, occupation, location, and many other factors.")

st.divider()
st.caption("Built with Streamlit • Data sources: IPUMS USA, "
           "University of Washington Center for Women's Welfare, BLS")