# ==========================================
# 🧠 Milestone 5: RUL Prediction Dashboard
# ==========================================
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="RUL Monitoring Dashboard", layout="wide")

st.title("📊 Remaining Useful Life (RUL) Monitoring Dashboard")
st.markdown("""
This dashboard shows the **health condition of machines** based on predicted RUL values.  
Upload your dataset or view the existing prediction results.
""")

# --------------------------------------------------------
# 1️⃣ File Upload / Load Existing
# --------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload prediction file (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    default_path = "Milestone5_predictions.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info("ℹ️ Showing existing prediction data (Milestone5_predictions.csv).")
    else:
        st.warning("⚠️ No data found! Please upload a predictions CSV.")
        st.stop()

# --------------------------------------------------------
# 2️⃣ Data Overview
# --------------------------------------------------------
st.subheader("📋 Data Preview")
st.dataframe(df.head(), use_container_width=True)

# Check required columns
required_cols = {"Cycle", "Predicted_RUL", "Alert_Level"}
if not required_cols.issubset(df.columns):
    st.error(f"❌ Missing required columns. Expected: {required_cols}")
    st.stop()

# --------------------------------------------------------
# 3️⃣ Alert Summary
# --------------------------------------------------------
st.subheader("🚨 Machine Status Summary")

alert_counts = df["Alert_Level"].value_counts().reindex(["NORMAL", "WARNING", "CRITICAL"], fill_value=0)
col1, col2, col3 = st.columns(3)
col1.metric("🟢 Normal", alert_counts["NORMAL"])
col2.metric("🟡 Warning", alert_counts["WARNING"])
col3.metric("🔴 Critical", alert_counts["CRITICAL"])

# --------------------------------------------------------
# 4️⃣ RUL Trend Visualization
# --------------------------------------------------------
st.subheader("📈 RUL Trend Over Time")

fig = px.line(
    df,
    x="Cycle",
    y="Predicted_RUL",
    color="Alert_Level",
    color_discrete_map={"NORMAL": "green", "WARNING": "orange", "CRITICAL": "red"},
    title="Remaining Useful Life Over Time",
    markers=True
)
fig.update_traces(line=dict(width=3))
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# 5️⃣ Interactive Filtering
# --------------------------------------------------------
st.subheader("🔍 Filter by Alert Level")
alert_filter = st.multiselect("Select alert levels:", df["Alert_Level"].unique(), default=df["Alert_Level"].unique())

filtered_df = df[df["Alert_Level"].isin(alert_filter)]
st.dataframe(filtered_df, use_container_width=True)

# --------------------------------------------------------
# 6️⃣ Save / Edit Section
# --------------------------------------------------------
st.subheader("💾 Save Edited Data")

if st.button("Save Updated Dataset"):
    filtered_df.to_csv("Updated_Machine_Status.csv", index=False)
    st.success("✅ File saved as Updated_Machine_Status.csv")

st.markdown("---")
st.caption("AI-Powered RUL Prediction Dashboard")
