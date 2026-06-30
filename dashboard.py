import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

from vision import (
    detect_image,
    detect_frame,
    draw_summary
)

from database import (
    create_database,
    save_detection,
    load_detections,
    clear_history,
    total_detections,
    unique_objects,
    most_detected_object
)

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="VisionAssist AI",
    page_icon="👁️",
    layout="wide"
)

create_database()

# ----------------------------
# LOAD DATABASE
# ----------------------------

df = load_detections()

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("👁️ VisionAssist AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "📷 Image Detection",

        "📹 Live Camera",

        "📋 Detection History",

        "📊 Analytics",

        "ℹ️ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("VisionAssist AI v5.0")

# ----------------------------
# DASHBOARD
# ----------------------------

if page == "🏠 Dashboard":

    st.title("👁️ VisionAssist AI")

    st.success("Welcome to VisionAssist AI v5.0 🚀")

    col1, col2, col3 = st.columns(3)

    col1.metric(

        "Total Detections",

        total_detections()

    )

    col2.metric(

        "Unique Objects",

        unique_objects()

    )

    col3.metric(

        "Top Object",

        most_detected_object()

    )

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
VisionAssist AI is an AI-powered computer vision application built using:

- YOLOv8
- Streamlit
- SQLite
- OpenCV
- Python

Features include:

✅ Real-time Object Detection

✅ Image Detection

✅ Detection History

✅ Analytics Dashboard

✅ Confidence Scores

✅ Object Counting

✅ CSV Export
""")# ----------------------------
# IMAGE DETECTION
# ----------------------------

elif page == "📷 Image Detection":

    st.title("📷 Image Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image = np.array(image)

        st.subheader("Original Image")

        st.image(image, use_container_width=True)

        if st.button("🚀 Detect Objects"):

            with st.spinner("Running YOLO Detection..."):

                annotated, counts, detections = detect_image(image)

            annotated = draw_summary(
                annotated,
                counts
            )

            st.success("Detection Complete!")

            st.subheader("Detected Image")

            st.image(
                annotated,
                use_container_width=True
            )

            st.subheader("Object Counts")

            st.json(counts)

            st.subheader("Confidence Scores")

            confidence_df = pd.DataFrame(detections)

            st.dataframe(
                confidence_df,
                use_container_width=True
            )

            for item in detections:

                save_detection(
                    item["object"],
                    item["confidence"]
                )

# ----------------------------
# LIVE CAMERA
# ----------------------------

elif page == "📹 Live Camera":

    st.title("📹 Live Camera")

    st.info("Real-Time YOLOv8 Detection")

    class YOLOProcessor(VideoProcessorBase):

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")

            annotated, counts, detections = detect_frame(img)

            annotated = draw_summary(
                annotated,
                counts
            )

            for item in detections:

                save_detection(
                    item["object"],
                    item["confidence"]
                )

            return annotated

    webrtc_streamer(
        key="visionassist",
        video_processor_factory=YOLOProcessor
    )# ----------------------------
# DETECTION HISTORY
# ----------------------------

elif page == "📋 Detection History":

    st.title("📋 Detection History")

    history = load_detections()

    if history.empty:

        st.warning("No detection history found.")

    else:

        st.dataframe(
            history,
            use_container_width=True
        )

        csv = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download CSV",
            csv,
            "detection_history.csv",
            "text/csv"
        )

        if st.button("🗑️ Clear History"):

            clear_history()

            st.success("History cleared.")

            st.rerun()

# ----------------------------
# ANALYTICS
# ----------------------------

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    history = load_detections()

    if history.empty:

        st.warning("No data available.")

    else:

        counts = history["object_name"].value_counts()

        st.subheader("Object Counts")

        fig, ax = plt.subplots(figsize=(8,4))

        counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Objects")

        ax.set_ylabel("Count")

        st.pyplot(fig)

        st.subheader("Distribution")

        fig2, ax2 = plt.subplots(figsize=(6,6))

        counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax2
        )

        ax2.set_ylabel("")

        st.pyplot(fig2)

# ----------------------------
# ABOUT
# ----------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ About VisionAssist AI")

    st.markdown("""

# 👁️ VisionAssist AI v5.0

An AI-powered Computer Vision application built using:

- Python
- Streamlit
- YOLOv8
- OpenCV
- SQLite
- Pandas
- Matplotlib

## Features

✅ Live Camera Detection

✅ Image Detection

✅ Detection History

✅ Analytics Dashboard

✅ Confidence Scores

✅ Object Counting

✅ CSV Export

---

### Developer

**Edwin**

Made with ❤️ using Python & AI.

""")

# ----------------------------
# FOOTER
# ----------------------------

st.markdown("---")

st.caption(
    "VisionAssist AI v5.0 | Developed by Edwin 🚀"
)
    
    