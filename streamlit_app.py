# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="🌊 Flood Risk Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌊 Flood Risk Prediction Dashboard")
st.markdown("This dashboard predicts **flood risk levels** based on historical and climatic factors.")

# ----------------------------
# LOAD MODEL & DATA
# ----------------------------
@st.cache_resource
def load_model():
    with open("best_flood_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Load processed dataset (used for visualizations)
df = pd.read_csv("flood_processed.csv")

# ----------------------------
# SIDEBAR - USER INPUT
# ----------------------------
st.sidebar.header("🔧 Input Parameters")
st.sidebar.write("Adjust the values below to predict flood risk.")

# Dynamically create input sliders based on dataset columns (except target)
feature_cols = [col for col in df.columns if col != "FloodProbability"]

user_input = {}
for col in feature_cols:
    user_input[col] = st.sidebar.slider(
        label=col,
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(df[col].mean()),
        step=0.1
    )

input_df = pd.DataFrame([user_input])

# ----------------------------
# PREDICTION
# ----------------------------
st.subheader("📌 Prediction Result")
prediction = model.predict(input_df)[0]

risk_map = {0: "🟢 Low Risk", 1: "🟠 Medium Risk", 2: "🔴 High Risk"}
st.metric(label="Flood Risk Level", value=risk_map[prediction])

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
st.subheader("📊 Feature Importance")
if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance (Random Forest)"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance not available for this model.")

# ----------------------------
# DATA VISUALIZATION
# ----------------------------
st.subheader("📈 Interactive Feature Visualization")
selected_feature = st.selectbox("Select a feature to visualize:", feature_cols)
fig = px.scatter(
    df, x=selected_feature, y="FloodProbability",
    color=pd.cut(df["FloodProbability"], bins=[-0.01, 0.33, 0.66, 1], labels=["Low", "Medium", "High"]),
    title=f"Flood Probability vs {selected_feature}",
    opacity=0.7
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# MODEL PERFORMANCE (OPTIONAL)
# ----------------------------
st.subheader("📊 Model Performance Metrics")

# Create dummy X, y for evaluation
X = df[feature_cols]
y_true = pd.cut(df["FloodProbability"], bins=[-0.01, 0.33, 0.66, 1], labels=[0, 1, 2])
y_pred = model.predict(X)

acc = accuracy_score(y_true, y_pred)
st.write(f"**✅ Accuracy:** {acc:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Classification Report
st.text("📋 Classification Report:")
st.text(classification_report(y_true, y_pred))

# ----------------------------
# DOWNLOAD PREDICTION
# ----------------------------
st.subheader("📥 Download Your Prediction")
output_df = input_df.copy()
output_df["Predicted_Flood_Risk"] = prediction
st.download_button(
    label="Download Prediction as CSV",
    data=output_df.to_csv(index=False),
    file_name="flood_risk_prediction.csv",
    mime="text/csv"
)

st.success("✅ Dashboard ready! Adjust inputs from the sidebar to generate predictions.")
