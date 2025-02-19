# Diabetes Prediction Model

This repository contains a **Streamlit web application** for predicting the likelihood of diabetes based on user input features. The application uses a deep learning model built with TensorFlow and Keras to make predictions. The dataset used for training and testing the model is the `diabetes_prediction_dataset.csv`.

**App Link**: [Diabetes Prediction App](https://diabaetespredictor993.streamlit.app/)

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
The **Diabetes Prediction Model** is a web application that allows users to input their health-related features and get a prediction about the likelihood of having diabetes. The model is trained on a dataset containing features such as age, BMI, blood glucose levels, and smoking history. The application is built using **Streamlit** and leverages a **deep learning model** for accurate predictions.

---

## Features
- **User Input**: Users can input their health-related features via a sidebar.
- **Exploratory Data Analysis (EDA)**: Visualize dataset insights such as missing values, gender distribution, and smoking history.
- **Model Training**: Train a deep learning model with early stopping to prevent overfitting.
- **Prediction**: Predict the probability of diabetes based on user input.
- **Interactive Visualizations**: Display model training results, including loss and accuracy curves.

---

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-model.git
   cd diabetes-prediction-model
