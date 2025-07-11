import pandas as pdimport numpy as npfrom sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import StandardScalerfrom sklearn.metrics import accuracy_score, classification_reportimport tensorflow as tffrom tensorflow.keras.models import Sequential, load_modelfrom tensorflow.keras.layers import Dense, Dropoutfrom tensorflow.keras.callbacks import EarlyStoppingimport keras_tuner as ktimport joblibimport shapimport streamlit as stimport matplotlib.pyplot as pltimport os
Function to train and save the model
def train_model():    # Load and preprocess the dataset    data = pd.read_csv('diabetes.csv')    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']    data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)    data[columns_with_zeros] = data[columns_with_zeros].fillna(data[columns_with_zeros].median())
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define hypermodel
def build_model(hp):
    model = Sequential()
    model.add(Dense(
        units=hp.Int('units_1', min_value=32, max_value=128, step=16),
        activation='relu',
        input_shape=(X_train.shape[1],)
    ))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'units_{i+2}', min_value=16, max_value=64, step=16),
            activation='relu'
        ))
        model.add(Dropout(hp.Float(f'dropout_{i+2}', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Perform hyperparameter tuning
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='tuner_dir',
    project_name='diabetes_tuning'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stopping])

# Get and train the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, 
               callbacks=[early_stopping], verbose=1)

# Evaluate the model
y_pred = (best_model.predict(X_test) > 0.5).astype(int)
st.write("Model Performance on Test Set:")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))

# Save model and scaler
best_model.save('diabetes_tensorflow_best_model.h5')
joblib.dump(scaler, 'scaler.pkl')
st.write("Model and scaler saved successfully.")

Streamlit app
def main():    st.title("Diabetes Prediction App")    st.write("Enter patient details to predict diabetes risk.")
# Check if model and scaler exist, else train the model
model_path = 'diabetes_tensorflow_best_model.h5'
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.write("Training model... This may take a few minutes.")
    train_model()

# Load model and scaler
model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Input fields for features
st.subheader("Patient Data Input")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=100.0)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=1000.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Prepare input data
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = (model.predict(input_data_scaled) > 0.5).astype(int)[0][0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.subheader("Prediction")
    st.write(f"The patient is predicted to be: **{result}**")
    
    # SHAP explanation
    st.subheader("Feature Importance (SHAP)")
    explainer = shap.DeepExplainer(model, scaler.transform(pd.read_csv('diabetes.csv').drop('Outcome', axis=1)[:100]))
    shap_values = explainer.shap_values(input_data_scaled)
    
    # Plot SHAP force plot
    shap.initjs()
    fig = shap.force_plot(explainer.expected_value[0], shap_values[0][0], input_data[0], 
                         feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                         matplotlib=True, show=False)
    st.pyplot(fig)

# Plot training history if model was trained
if os.path.exists(model_path):
    st.subheader("Training History")
    history = model.fit(scaler.transform(pd.read_csv('diabetes.csv').drop('Outcome', axis=1)[:100]), 
                       pd.read_csv('diabetes.csv')['Outcome'][:100], epochs=1, verbose=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig)

if name == "main":    main()
