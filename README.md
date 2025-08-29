Credit Card Fraud Detection System
📌 Overview
This project is focused on detecting fraudulent transactions using Machine Learning techniques.
The dataset used is highly imbalanced, where fraudulent transactions are much less compared to legitimate ones.
Our goal is to build models that can correctly identify fraud with high accuracy and precision.

🛠 Features
✅ Data Preprocessing and Cleaning

✅ Handling imbalanced dataset using SMOTE (Synthetic Minority Oversampling Technique)

✅ Multiple ML Models Implemented:

Logistic Regression

Random Forest

XGBoost (optional)

✅ Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC Curve

✅ Interactive Streamlit Dashboard for visualization and predictions


Installation
Clone this repository

git clone https://github.com/ctcrahul/Credit_Fraudd_Detection_System



Install dependencies


pip install -r requirements.txt
Run the Streamlit app


streamlit run app.py


📊 Results

Logistic Regression: ~94% Accuracy

Random Forest: ~99% Accuracy

ROC-AUC Score: 0.98+

Visualizations include:

Fraud vs Non-Fraud Distribution

Confusion Matrix

ROC Curve

🌐 Deployment
This project is deployed on Streamlit Cloud for easy access.
👉 https://credit-fraud-system.streamlit.app/

