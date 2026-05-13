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

    # Field of degree — only meaningful for Bachelor's+
    if educ >= 11:
        degfield_options = [
            ("Computer / Information Sciences", 21),
            ("Engineering",                     24),
            ("Business / Finance / Accounting", 62),
            ("Health / Nursing / Medical",      61),
            ("Education / Teaching",            23),
            ("Psychology",                      52),
            ("Social Sciences (Econ, Soc)",     55),
            ("Mathematics / Statistics",        37),
            ("Life Sciences (Biology)",         36),
            ("Physical Sciences (Chem, Phys)",  50),
            ("Liberal Arts / Humanities",       34),
            ("English / Literature",            33),
            ("Communications",                  19),
            ("Fine Arts",                       60),
            ("History",                         64),
            ("Public Affairs / Social Work",    54),
            ("Other / Interdisciplinary",       40),
        ]
        degfield_choice = st.selectbox(
            "Field of degree",
            degfield_options,
            index=0,
            format_func=lambda x: x[0]
        )
        degfield = degfield_choice[1]
    else:
        degfield = 0

    # ---- Immigration / origin questions ----
    born_in_usa = st.radio(
        "Were you born in the USA?",
        ["Yes", "No"],
        horizontal=True,
    )

    if born_in_usa == "Yes":
        foreign_born = 0
        citizen = 0
        yrsusa1 = 0
        speakeng = 3
    else:
        foreign_born = 1
        yrsusa1 = st.slider("Years living in the USA", 0, 60, 10)
        citizen_options = [
            (3, "Not a U.S. citizen"),
            (2, "Naturalized U.S. citizen"),
            (1, "Born abroad of American parents"),
        ]
        citizen = st.selectbox("Citizenship status", citizen_options,
                               format_func=lambda x: x[1])[0]
        speakeng_options = [
            (3, "Only English"),
            (4, "Very well"),
            (5, "Well"),
            (6, "Not well"),
            (1, "Does not speak English"),
        ]
        speakeng = st.selectbox("English proficiency", speakeng_options,
                                index=1,
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

    occupation_options = [
        ("Management / executive",          110,  7860),
        ("Business / finance",              800,  6870),
        ("Computer / IT",                   1010, 6470),
        ("Engineering / architecture",      1310, 7860),
        ("Science / research",              1600, 7860),
        ("Education (teaching)",            2310, 7860),
        ("Healthcare practitioner",         3010, 7970),
        ("Healthcare support",              3600, 7970),
        ("Service (food, personal, security)", 4000, 8680),
        ("Sales / office / admin",          4700, 5380),
        ("Construction / trades",           6200, 770),
        ("Production / transportation",     7700, 6070),
    ]
    occ_choice = st.selectbox(
        "Occupation",
        occupation_options,
        index=9,
        format_func=lambda x: x[0]
    )
    occ = occ_choice[1]
    ind = occ_choice[2]

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
user_inputs = {
    "AGE": age,
    "SEX": sex,
    "RACE": 1,
    "HISPAN": 0,
    "MARST": 1 if adults >= 2 else 6,
    "FAMSIZE": adults + infants + preschoolers + schoolagers + teenagers,
    "NCHILD": infants + preschoolers + schoolagers + teenagers,
    "NCHLT5": infants + preschoolers,
    "EDUC": educ,
    "DEGFIELD": degfield,
    "OCC": occ,
    "IND": ind,
    "CLASSWKR": classwkr,
    "UHRSWORK": uhrswork,
    "WKSWORK2": wkswork2,
    "PUMA": 1101,
    "FOREIGN_BORN": foreign_born,
    "CITIZEN": citizen,
    "SPEAKENG": speakeng,
    "YRSUSA1": yrsusa1,
}

# ============================================================
# PAGE 2: PREDICTION
# ============================================================
st.header("2. Your Predicted Earned Income")

# Detect "not currently working" case — model wasn't trained for this
is_unemployed = (uhrswork == 0)

if is_unemployed:
    st.warning(
        "⚠️ **You indicated 0 hours worked per week.** This model predicts "
        "earned income from work, so it can't generate a meaningful income "
        "estimate for someone not currently employed. Below, we show what "
        "someone with your characteristics could expect to earn at "
        "full-time hours (40 hrs/week, 50–52 weeks/year) — a goal "
        "projection, not a current prediction."
    )
    employed_inputs = user_inputs.copy()
    employed_inputs["UHRSWORK"] = 40
    employed_inputs["WKSWORK2"] = 6
    predicted_income = predict_income(model, features, employed_inputs)
    hourly_divisor = 40 * 52
    income_label = "Projected at full-time"
else:
    predicted_income = predict_income(model, features, user_inputs)
    hourly_divisor = uhrswork * 52
    income_label = "Predicted annual income"

m1, m2, m3 = st.columns(3)
with m1:
    st.metric(income_label, f"${predicted_income:,.0f}")
with m2:
    st.metric("Per month", f"${predicted_income/12:,.0f}")
with m3:
    st.metric("Hourly equivalent", f"${predicted_income/hourly_divisor:,.2f}")

st.caption("Based on an XGBoost model trained on Maryland workers "
           "(IPUMS USA 2023 ACS, ages 18–64). Test MAE: $19,622. R² = 0.615.")

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
    # Special handling: unemployed users have one priority, not a list
    if is_unemployed:
        st.info(
            "**Your most impactful action: find employment.** Based on your "
            f"characteristics, working full-time at 40 hours/week, 50–52 "
            f"weeks/year could bring your income to "
            f"**${predicted_income:,.0f}/year**."
        )
        st.write("**Additional improvements** once employed:")
    else:
        st.write(
            "**What if you changed one thing?** These scenarios re-run the "
            "model with one input changed to show potential impact:"
        )

    scenarios = []

    # Scenario 1: full-time hours (skip if unemployed — already handled above)
    if uhrswork < 40 and not is_unemployed:
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

    # Scenario 5: switch to a higher-paying occupation
    occupation_swaps = [
        ("Management / executive",       110,  7860),
        ("Computer / IT",                1010, 6470),
        ("Healthcare practitioner",      3010, 7970),
    ]
    for label, alt_occ, alt_ind in occupation_swaps:
        if alt_occ != user_inputs["OCC"]:
            new_inputs = user_inputs.copy()
            new_inputs["OCC"] = alt_occ
            new_inputs["IND"] = alt_ind
            new_pred = predict_income(model, features, new_inputs)
            delta = new_pred - predicted_income
            if delta > 5000:
                scenarios.append((f"Switch to {label}", new_pred, delta))

    # Scenario 6: Retrain in a higher-paying field (only if bachelor's+)
    if educ >= 11:
        field_swaps = [
            ("Computer / Information Sciences", 21,  1010, 6470),
            ("Engineering",                     24,  1310, 7860),
            ("Health / Nursing / Medical",      61,  3010, 7970),
        ]
        for label, alt_field, alt_occ, alt_ind in field_swaps:
            if alt_field != user_inputs["DEGFIELD"]:
                new_inputs = user_inputs.copy()
                new_inputs["DEGFIELD"] = alt_field
                new_inputs["OCC"] = alt_occ
                new_inputs["IND"] = alt_ind
                new_pred = predict_income(model, features, new_inputs)
                delta = new_pred - predicted_income
                if delta > 10000:
                    scenarios.append(
                        (f"Retrain: degree in {label} + matching job",
                         new_pred, delta)
                    )

    if scenarios:
        for label, new_income, delta in scenarios:
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.write(f"**{label}**")
            with c2:
                st.write(f"New income: ${new_income:,.0f}")
            with c3:
                color = "🟢" if delta > 0 else "🔴"
                st.write(
                    f"{color} {'+' if delta>=0 else ''}${delta:,.0f}/year"
                )
    else:
        st.write("You're already at the highest values for these factors.")

    st.caption(
        "**About these scenarios:** Recommendations show statistical "
        "averages from Maryland workers — what people with similar "
        "characteristics typically earn after such a change. They are not "
        "career advice. Real transitions require time and investment: "
        "an occupation switch may take months of job-searching or training; "
        "retraining with a new degree typically requires 2–4 years and "
        "tuition costs. The model identifies which changes would *most* "
        "improve income on average; weighing them against your personal "
        "situation is up to you."
    )

st.divider()
st.caption("Built with Streamlit • Data sources: IPUMS USA, "
           "University of Washington Center for Women's Welfare, BLS")