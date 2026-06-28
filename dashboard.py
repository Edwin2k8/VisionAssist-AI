import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="VisionAssist AI",
    page_icon="👁️",
    layout="wide"
)

# ---------------- LOAD DATABASE ----------------

conn = sqlite3.connect("detections.db")
df = pd.read_sql_query("SELECT * FROM detections", conn)
conn.close()

# ---------------- SIDEBAR ----------------

st.sidebar.title("👁️ VisionAssist AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📋 Detection History",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

# ---------------- DASHBOARD ----------------

if page == "🏠 Dashboard":

    st.title("👁️ VisionAssist AI Dashboard")

    st.success("Welcome to VisionAssist AI Dashboard 🚀")

    total_detections = len(df)
    unique_objects = df["object_name"].nunique()

    if total_detections > 0:
        top_object = df["object_name"].value_counts().idxmax()
    else:
        top_object = "None"

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Detections", total_detections)
    col2.metric("Unique Objects", unique_objects)
    col3.metric("Top Object", top_object)

# ---------------- DETECTION HISTORY ----------------

elif page == "📋 Detection History":

    st.title("📋 Detection History")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)

    st.download_button(
        "📥 Download Detection History",
        csv,
        file_name="detection_history.csv",
        mime="text/csv"
    )

# ---------------- ANALYTICS ----------------

elif page == "📊 Analytics":

    st.title("📊 Analytics")

    if len(df) > 0:

        object_counts = df["object_name"].value_counts()

        st.subheader("📊 Most Detected Objects")

        fig, ax = plt.subplots()

        object_counts.plot(kind="bar", ax=ax)

        ax.set_xlabel("Object")
        ax.set_ylabel("Count")

        st.pyplot(fig)

        st.subheader("🥧 Object Distribution")

        fig2, ax2 = plt.subplots()

        object_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax2
        )

        ax2.set_ylabel("")

        st.pyplot(fig2)

    else:
        st.info("No detection data available.")

# ---------------- ABOUT ----------------

elif page == "ℹ️ About":

    st.title("ℹ️ About VisionAssist AI")

    st.markdown("""
## 👁️ VisionAssist AI

VisionAssist AI is a Computer Vision analytics dashboard built using Python, Streamlit, SQLite, Pandas and Matplotlib.

### Features

- 📊 Detection Analytics
- 📋 Detection History
- 📥 CSV Export
- 📈 Charts & Statistics

### Technologies

- Python
- Streamlit
- SQLite
- Pandas
- Matplotlib

### Developer

Edwin
""")
