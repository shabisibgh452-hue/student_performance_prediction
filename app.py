import streamlit as st
import joblib
import numpy as np

# Load Trained Model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

st.title("🎓 Student Performance Prediction")
st.write("Enter the student's details and click Predict.")

# ====== Input Fields ======
study_hours = st.number_input("Study Hours", min_value=0.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.5)
internet_usage = st.number_input("Internet Usage (Hours)", min_value=0.0, step=0.5)
assignments_completed = st.number_input("Assignments Completed", min_value=0)
previous_score = st.number_input("Previous Score", min_value=0.0)
placement_status = st.selectbox("Placement Status", ["No", "Yes"])

# Convert Placement Status
placement = 1 if placement_status == "Yes" else 0

# ====== Prediction ======
if st.button("Predict Exam Score"):

    input_data = np.array([[study_hours,
                            attendance,
                            sleep_hours,
                            internet_usage,
                            assignments_completed,
                            previous_score,
                            placement]])

    prediction = model.predict(input_data)

    st.success(f"🎯 Predicted Exam Score: {prediction[0]:.2f}")