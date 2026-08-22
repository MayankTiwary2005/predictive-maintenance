# ⚙ Predictive Maintenance System

A machine learning based predictive maintenance system that predicts the probability of machine failure using real-time operating parameters.

## 🚀 Live Demo

Coming soon.

## 📌 Project Overview

This project uses machine operating parameters to predict whether a machine is likely to experience failure.

The dataset used is the AI4I 2020 Predictive Maintenance Dataset.

The original project experimented with an LSTM-based approach. During model evaluation, traditional machine learning models were compared, and Random Forest provided significantly better performance for this dataset.

## 🔧 Input Features

The model uses five machine parameters:

- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

## 🤖 Models Compared

- LSTM
- Logistic Regression
- Random Forest
- XGBoost

## 🏆 Final Model

Random Forest Classifier

The Random Forest model achieved:

- ROC-AUC: 0.9681
- PR-AUC: 0.7343
- Accuracy: 0.9831
- Precision: 0.5909
- Recall: 0.7429
- F1 Score: 0.6582

Because machine failure is highly imbalanced, the project focuses on metrics such as Recall, F1-score, ROC-AUC and PR-AUC rather than accuracy alone.

## 📊 Dataset

AI4I 2020 Predictive Maintenance Dataset.

Total samples: 10,000

Machine failure distribution:

- Normal: 9,661
- Failure: 339

Failure rate: 3.39%

## 🌐 Web Application

The project includes an interactive Streamlit application where users can enter:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The application returns:

- Failure probability
- Predicted machine condition
- Model confidence visualization
- Feature importance
- Training-distribution warnings

## 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Streamlit
- Plotly
- Joblib

## 📁 Project Structure

```text
predictive-maintenance/
│
├── app.py
├── random_forest_model.pkl
├── feature_names.pkl
├── threshold.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── notebook/
    └── predictive_maintenance.ipynb