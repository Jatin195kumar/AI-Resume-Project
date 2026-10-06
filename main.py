import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Resume Screening ML",
    page_icon="📄",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("📄 Resume Screening ML")
st.write("Machine Learning prediction demo")

st.info(
    "This application is an educational/portfolio demonstration "
    "of a machine-learning classification model."
)

# -----------------------------
# Input fields
# -----------------------------
st.header("Enter Candidate Information")

years_experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.5
)

skills_match_score = st.number_input(
    "Skills Match Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

education_level = st.selectbox(
    "Education Level",
    ["Bachelor", "Master", "PhD"]
)

project_count = st.number_input(
    "Number of Projects",
    min_value=0,
    max_value=50,
    value=2,
    step=1
)

resume_length = st.number_input(
    "Resume Length",
    min_value=0,
    max_value=10000,
    value=500,
    step=50
)

github_activity = st.number_input(
    "GitHub Activity",
    min_value=0,
    max_value=1000,
    value=10,
    step=1
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict", type="primary"):

    input_data = pd.DataFrame({
        "years_experience": [years_experience],
        "skills_match_score": [skills_match_score],
        "education_level": [education_level],
        "project_count": [project_count],
        "resume_length": [resume_length],
        "github_activity": [github_activity]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Convert prediction to original label
    if prediction == 1:
        result = "Yes"
    else:
        result = "No"

    st.subheader("Prediction")

    if result == "Yes":
        st.success("Model prediction: Yes")
    else:
        st.warning("Model prediction: No")

    st.write(
        f"Model probability for **Yes**: "
        f"{probability[1] * 100:.2f}%"
    )

    st.write(
        f"Model probability for **No**: "
        f"{probability[0] * 100:.2f}%"
    )

    st.subheader("Input Data")
    st.dataframe(input_data, use_container_width=True)