import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your creditcard.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.write("### 📋 Dataset Preview")
    st.dataframe(data.head())

    if "Class" in data.columns:
        X = data.drop("Class", axis=1)
        y_true = data["Class"]
    else:
        X = data.copy()
        y_true = None

    if "Time" in X.columns:
        X = X.drop("Time", axis=1)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    data["Fraud_Prediction"] = y_pred
    data["Fraud_Probability"] = y_prob

    st.subheader("📊 Prediction Summary")
    total = len(data)
    fraud = int(sum(y_pred))
    legit = total - fraud

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Fraud", fraud)
    col3.metric("Legit", legit)

    st.subheader("📈 Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.pie([legit, fraud], labels=["Legit", "Fraud"], autopct="%1.1f%%", startangle=90, colors=["#36A2EB", "#FF6384"])
        ax1.axis("equal")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.histplot(data=data, x="Amount", hue="Fraud_Prediction", bins=40, ax=ax2)
        st.pyplot(fig2)

    if y_true is not None:
        st.subheader("📉 Model Evaluation")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        cm = confusion_matrix(y_true, y_pred)
        fig3, ax3 = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
        disp.plot(ax=ax3, cmap="Blues", colorbar=False)
        st.pyplot(fig3)

        roc = roc_auc_score(y_true, y_prob)
        st.metric("ROC-AUC", round(roc, 3))

    st.subheader("📥 Download Predictions")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "fraud_predictions.csv", "text/csv")

else:
    st.info("⬅️ Please upload your creditcard1.csv file to begin.")
