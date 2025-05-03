import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("medical_insurance.csv")

# --- Page Config ---
st.set_page_config(page_title="ML-Finance Explorer", layout="wide", page_icon="📈")

# --- Welcome Section ---
st.title("📊 ML-Finance Explorer")
st.markdown("Welcome to your Machine Learning on Financial Data app! 🧠")

st.image("https://media.giphy.com/media/Ll22OhMLAlVDb8UQWe/giphy.gif", width=400)

# --- Sidebar ---
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your Kragle Dataset (.csv)", type=["csv"])
yahoo_data_checkbox = st.sidebar.checkbox("Fetch from Yahoo Finance (not used in this app)")

# --- Global Variables ---
df = None
model = None
X_train = X_test = y_train = y_test = y_pred = None

# --- Step 1: Load Data ---
if st.button("🔍 Load Data"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded successfully!")
        st.dataframe(df.head())
    else:
        st.error("❌ Please upload a dataset.")

# --- Step 2: Preprocessing ---
if st.button("🧹 Preprocess Data"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.info("Checking for missing values...")
        st.write(df.isnull().sum())

        st.success("✅ Data preprocessing completed!")
    else:
        st.error("❌ Please load the dataset first.")

# --- Step 3: Feature Engineering ---
if st.button("🛠️ Feature Engineering"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = pd.get_dummies(df, drop_first=True)

        st.write("Transformed Data Sample:")
        st.dataframe(df.head())

        st.success("✅ Feature engineering completed!")
    else:
        st.error("❌ Please load the dataset first.")

# --- Step 4: Train/Test Split ---
if st.button("📊 Train/Test Split"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = pd.get_dummies(df, drop_first=True)

        X = df.drop("charges", axis=1)
        y = df["charges"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        fig = px.pie(names=["Train", "Test"], values=[len(X_train), len(X_test)], title="Train/Test Split")
        st.plotly_chart(fig)
        st.success("✅ Data split successfully!")
    else:
        st.error("❌ Please load the dataset first.")

# --- Step 5: Train Model ---
if st.button("🤖 Train Model"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = pd.get_dummies(df, drop_first=True)
        X = df.drop("charges", axis=1)
        y = df["charges"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        st.success("✅ Linear Regression model trained successfully!")
    else:
        st.error("❌ Please load the dataset first.")

# --- Step 6: Evaluate Model ---
if st.button("📈 Evaluate Model"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = pd.get_dummies(df, drop_first=True)
        X = df.drop("charges", axis=1)
        y = df["charges"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write(f"**MAE**: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"**MSE**: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"**R² Score**: {r2_score(y_test, y_pred):.2f}")

        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual Charges", "y": "Predicted Charges"}, title="Actual vs Predicted")
        st.plotly_chart(fig)
    else:
        st.error("❌ Please load the dataset first.")

# --- Final Note ---
st.markdown("---")
st.info("Made with ❤️ for AF3005 – Programming for Finance")
