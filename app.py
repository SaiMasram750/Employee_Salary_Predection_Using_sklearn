import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("./best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (must match training features)
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Others"
])
fnlwgt = st.sidebar.number_input("Fnlwgt", 10000, 1000000, 50000)
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
capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 99999, 0)
educational_num = st.sidebar.slider("Educational Num", 5, 16, 10)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "England", "Others"
])

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
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    # Ensure columns match
    required_cols = [
        'age', 'workclass', 'fnlwgt', 'marital-status', 'occupation', 'relationship',
        'race', 'gender', 'capital-gain', 'capital-loss', 'educational-num',
        'hours-per-week', 'native-country'
    ]
    missing_cols = [col for col in required_cols if col not in batch_data.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {missing_cols}")
    else:
        batch_preds = model.predict(batch_data[required_cols])
        batch_data['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.write(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name="predictions.csv", mime="text/csv")
