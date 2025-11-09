import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained pipeline (preprocessing + model)
MODEL_PATH = "best_model.pkl"
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
else:
    st.error("❌ Model file not found. Please ensure 'best_model.pkl' is in the app directory.")
    st.stop()

# Page setup
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Others"
])
fnlwgt = st.sidebar.number_input("Fnlwgt", min_value=10000, max_value=1000000, value=50000)
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
    "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
    "Priv-house-serv", "Protective-serv", "Armed-Forces", "Others"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
educational_num = st.sidebar.slider("Educational Num", 5, 16, 10)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "England", "Others"
])

# Create input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'educational-num': [educational_num],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.dataframe(input_df)

# Predict button
if st.button("Predict Salary Class"):
    try:
        prediction = pipeline.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(batch_data.head())

        predictions = pipeline.predict(batch_data)
        batch_data['PredictedClass'] = predictions
        st.write("✅ Predictions:")
        st.dataframe(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
