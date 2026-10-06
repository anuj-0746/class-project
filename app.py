import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup model path dynamically
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl" if (BASE_DIR / "best_model.pkl").exists() else BASE_DIR / "models" / "best_model.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

st.title("Classroom Exam Score Predictor")
st.write("Enter your habits and past academic data below to receive your personalized exam forecast.")

# Inputs matching the 5 features expected by the trained model:
# 1. Attendance (%)
# 2. Daily Study Hours
# 3. Daily Screen Time (Hours)
# 4. Daily Sleep (Hours)
# 5. Daily Exercise (Hours)

attendance_percent = st.number_input("Current Attendance Rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
daily_study_hours = st.number_input("Average Study Hours per day", min_value=0.0, max_value=24.0, value=2.5, step=0.5)
daily_screen_time = st.number_input("Average Screen Time per day (Hours)", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
daily_sleep_hours = st.number_input("Average Sleep Hours per night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
daily_exercise_hours = st.number_input("Average Exercise Hours per day", min_value=0.0, max_value=24.0, value=1.0, step=0.5)

if st.button("Predict My Score"):
    # Create DataFrame with exact column names expected by scikit-learn model
    input_df = pd.DataFrame([{
        'Attendance (%)': attendance_percent,
        'Daily Study Hours': daily_study_hours,
        'Daily Screen Time (Hours)': daily_screen_time,
        'Daily Sleep (Hours)': daily_sleep_hours,
        'Daily Exercise (Hours)': daily_exercise_hours
    }])
    
    # Generate prediction
    prediction = model.predict(input_df)[0]
    
    # Clip limits to keep score within realistic bounds (0 - 100)
    prediction = max(0.0, min(100.0, float(prediction)))
    
    st.success(f"Your Predicted Exam Score: {prediction:.1f}%")
    
    # AI Disclaimer to encourage students
    st.info("**Note:** This is a statistical forecast based on class trends. "
            "If you want to beat the algorithm, you can change your daily habits starting today!")