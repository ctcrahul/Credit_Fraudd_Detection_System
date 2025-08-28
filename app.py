# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection with Logistic Regression + SMOTE ")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.write(dataset.head())

    # Scale Time and Amount
    scaler = RobustScaler()
    dataset[["Time", "Amount"]] = scaler.fit_transform(dataset[["Time", "Amount"]])

    # Features and target
    X = dataset.drop("Class", axis=1)
    y = dataset["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optional: use a subset for faster demo
    if st.checkbox("Use small sample for fast demo (~10% data)"):
        X_train, y_train = X_train.sample(frac=0.1, random_state=42), y_train.loc[X_train.index]

    # SMOTE oversampling
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    st.info(f"Training set size after SMOTE: {len(X_train_res)} samples")

    # Logistic Regression
    logreg = LogisticRegression(
        penalty='l1',
        C=0.1,
        solver='liblinear',
        n_jobs=-1,
        random_state=42
    )
    logreg.fit(X_train_res, y_train_res)
    y_pred = logreg.predict(X_test)
    y_proba = logreg.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("Model Performance Metrics ")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve ")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC={roc_auc:.3f})")
    ax.plot([0,1], [0,1], color='orange', linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc='lower right')
    st.pyplot(fig)

    st.success("All done! You can interact with the checkbox to speed up demo.")
else:
    st.info("Please upload a CSV file to begin.")
