# app.py (Streamlit UI)
import streamlit as st
import os
from tracker import process_video

st.set_page_config(page_title="Video Tracking Demo", layout="centered")
st.title("Video Tracking Demo — YOLO-Seg + ByteTrack")

st.markdown(
    """
    Upload a video (mp4/avi). The app will run YOLO-Seg + ByteTrack (if installed) to produce:
    - `results.json` (frame-wise tracked objects)
    - `annotated_video.mp4` (video with boxes + IDs)
    """
)

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
model_path = st.text_input("Model path (relative to repo)", "models/best.pt")
conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25)

if uploaded is not None:
    os.makedirs("uploads", exist_ok=True)
    vid_path = os.path.join("uploads", uploaded.name)
    with open(vid_path, "wb") as f:
        f.write(uploaded.read())

    st.video(vid_path)

    if st.button("Run tracking"):
        with st.spinner("Running inference and tracking — this can take a few minutes"):
            try:
                json_path, annotated_path = process_video(
                    vid_path,
                    model_path=model_path,
                    output_dir="outputs",
                    conf=conf
                )
                st.success("Processing complete ✅")
                st.video(annotated_path)

                with open(json_path, "r") as fh:
                    st.download_button("Download results.json", fh, file_name="results.json")

                with open(annotated_path, "rb") as fh:
                    st.download_button("Download annotated video", fh, file_name="annotated_video.mp4")
            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.write("See README for instructions to enable full ByteTrack support.")
