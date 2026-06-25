from ultralytics import YOLO
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer

model = YOLO("yolov8n.pt")

class YOLOProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img, imgsz=320, verbose=False)

        annotated_frame = results[0].plot()

        return annotated_frame

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
        "📹 Live Camera",
        "📋 Detection History",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

# ---------------- DASHBOARD ----------------

if page == "🏠 Dashboard":

    st.title("👁️ VisionAssist AI Dashboard")

    st.success("Welcome to VisionAssist AI v3.1 🚀")

    total_detections = len(df)

    unique_objects = df["object_name"].nunique()

    top_object = (
        df["object_name"]
        .value_counts()
        .idxmax()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Detections",
        total_detections
    )

    col2.metric(
        "Unique Objects",
        unique_objects
    )

    col3.metric(
        "Top Object",
        top_object
    )
# ---------------- LIVE CAMERA ----------------

elif page == "📹 Live Camera":

    st.title("📹 VisionAssist AI Live Detection")

    st.info("YOLOv8 Real-Time Object Detection")

    webrtc_streamer(
        key="yolo",
        video_processor_factory=YOLOProcessor
    )

# ---------------- DETECTION HISTORY ----------------

elif page == "📋 Detection History":

    st.title("📋 Detection History")

    st.dataframe(
        df,
        use_container_width=True
    )

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

    object_counts = (
        df["object_name"]
        .value_counts()
    )

    st.subheader("📊 Most Detected Objects")

    fig, ax = plt.subplots()

    object_counts.plot(
        kind="bar",
        ax=ax
    )

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

# ---------------- ABOUT ----------------

elif page == "ℹ️ About":

    st.title("ℹ️ About VisionAssist AI")

    st.markdown("""
    ## 👁️ VisionAssist AI

    VisionAssist AI is a Computer Vision application developed using Python, OpenCV, YOLOv8, SQLite, and Streamlit.

    ### Technologies Used

    - Python
    - OpenCV
    - YOLOv8
    - SQLite
    - Streamlit
    - Pandas
    - Matplotlib

    ### Features

    ✅ Face Detection

    ✅ Object Detection

    ✅ Object Counting

    ✅ Detection History

    ✅ Analytics Dashboard

    ✅ CSV Export

    ### Developer

    Edwin

    ### Future Improvements

    - Real-time Streamlit Webcam
    - Screenshot Capture
    - Object Confidence Scores
    - Dark Theme UI
    - Cloud Deployment
    """)

# ---------------- FOOTER ----------------

st.markdown("---")
st.caption("Developed by Edwin | VisionAssist AI v3.1")