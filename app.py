import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_prediction_dataset.csv')

data_frame = load_data()

# Streamlit App
st.title("Diabetes Prediction Model")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Display dataset
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data_frame)

# Exploratory Data Analysis (EDA)
st.subheader("Exploratory Data Analysis (EDA)")

if st.sidebar.checkbox("Show EDA"):
    st.write("First 5 rows of the dataset:")
    st.write(data_frame.head(5))

    st.write("Random 5 rows of the dataset:")
    st.write(data_frame.sample(5))

    st.write("Missing values in the dataset:")
    st.write(data_frame.isnull().sum())

    st.write("Dataset Info:")
    st.write(data_frame.info())

    st.write("Bar plot of Smoking History:")
    fig, ax = plt.subplots()
    data_frame.groupby('smoking_history').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
    plt.gca().spines[['top', 'right',]].set_visible(False)
    st.pyplot(fig)

    st.write("Bar plot of Gender:")
    fig, ax = plt.subplots()
    data_frame.groupby('gender').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
    plt.gca().spines[['top', 'right',]].set_visible(False)
    st.pyplot(fig)

# One Hot Encoding
data_frame = pd.get_dummies(data_frame, columns=['gender', 'smoking_history'], drop_first=True)

# Input/Output Features
X = data_frame.drop(columns=['diabetes'])
y = data_frame['diabetes']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train ANN Model
model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=X_train.shape[1]))
BatchNormalization()
model.add(Dense(units=128, activation='relu'))
BatchNormalization()
model.add(Dense(units=64, activation='relu'))
BatchNormalization()
model.add(Dense(units=32, activation='relu'))
BatchNormalization()
Dropout(0.2)
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping Method
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

# Model Training
if st.sidebar.checkbox("Train Model"):
    history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])

    st.subheader("Model Training Results")

    st.write("Model Loss:")
    fig, ax = plt.subplots()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    st.pyplot(fig)

    st.write("Model Accuracy:")
    fig, ax = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    st.pyplot(fig)

    # Accuracy Score
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction.round())
    st.write(f"Model Accuracy: {accuracy:.2f}")

# User Input for Prediction
st.sidebar.subheader("Make a Prediction")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.sidebar.slider("Age", 0, 100, 25)
    hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
    smoking_history = st.sidebar.selectbox("Smoking History", ["never", "former", "current", "not current"])
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    HbA1c_level = st.sidebar.slider("HbA1c Level", 3.5, 9.0, 5.0)
    blood_glucose_level = st.sidebar.slider("Blood Glucose Level", 80, 300, 100)

    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# One Hot Encoding for user input
input_df = pd.get_dummies(input_df, columns=['gender', 'smoking_history'], drop_first=True)

# Ensure the input dataframe has the same columns as the training data
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Feature Scaling for user input
input_scaled = scaler.transform(input_df)

# Make Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_scaled)
    st.subheader("Prediction")
    st.write("Probability of having diabetes:", prediction[0][0])
    st.write("Predicted Class:", "Diabetic" if prediction[0][0] > 0.5 else "Not Diabetic")
