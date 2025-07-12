import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# Title and description
st.title("Diabetes Prediction App")
st.markdown("""
This app uses a trained neural network to predict the likelihood of diabetes based on input features.
Enter the patient details below and click 'Predict' to see the results.
""")

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model('diabetes_tensorflow_best_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Please ensure 'diabetes_tensorflow_best_model.h5' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Define feature names (based on the diabetes dataset)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Create input fields for user data
st.subheader("Enter Patient Details")
cols = st.columns(4)
user_input = {}
for i, feature in enumerate(feature_names):
    with cols[i % 4]:
        if feature == 'Pregnancies':
            user_input[feature] = st.number_input(feature, min_value=0, max_value=20, value=0, step=1)
        elif feature == 'Age':
            user_input[feature] = st.number_input(feature, min_value=0, max_value=120, value=30, step=1)
        elif feature in ['Glucose', 'BloodPressure', 'Insulin']:
            user_input[feature] = st.number_input(feature, min_value=0.0, max_value=500.0, value=100.0, step=0.1)
        else:
            user_input[feature] = st.number_input(feature, min_value=0.0, max_value=100.0, value=0.0, step=0.1)

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input], columns=feature_names)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("Predict"):
    # Make prediction
    prediction_proba = model.predict(input_scaled)[0][0]
    prediction = 1 if prediction_proba > 0.5 else 0
    probability = prediction_proba * 100

    # Display prediction
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"The model predicts a **high risk of diabetes** with {probability:.2f}% probability.")
    else:
        st.success(f"The model predicts a **low risk of diabetes** with {100 - probability:.2f}% probability.")

    # SHAP explanation
    st.subheader("Feature Importance (SHAP)")
    explainer = shap.DeepExplainer(model, scaler.transform(pd.read_csv('diabetes.csv').drop('Outcome', axis=1).iloc[:100]))
    shap_values = explainer.shap_values(input_scaled)
    
    # Create SHAP summary plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[0], input_scaled, feature_names=feature_names, show=False)
    st.pyplot(fig)
    plt.close()

# Optional: Display model performance (requires test data)
st.subheader("Model Performance")
if st.checkbox("Show model performance metrics"):
    # Load test data (assuming diabetes.csv is available)
    try:
        data = pd.read_csv('diabetes.csv')
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        from sklearn.model_selection import train_test_split
        _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        st.write(f"**Test Accuracy**: {accuracy:.2f}")
        st.write("**Classification Report**:")
        st.json(report)
    except FileNotFoundError:
        st.warning("Cannot display performance metrics: 'diabetes.csv' not found.")

st.markdown("---")
st.markdown("Built with Streamlit and TensorFlow. Model trained on the Pima Indians Diabetes Dataset.")
