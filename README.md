Vehicle and Pedestrian Detection & Tracking 🚗🚶‍♀️

## 🚀 Live Demo
Try the deployed app here: [Streamlit Demo Link](https://vehicle-pedestrian-tracking-e2njc6uozbjkuhup6jvzxy.streamlit.app)

https://vehicle-pedestrian-tracking-e2njc6uozbjkuhup6jvzxy.streamlit.app/
📌 Project Overview

This project implements a YOLOv8 segmentation model combined with ByteTrack for detecting and tracking vehicles and pedestrians in videos.

The workflow includes:

Collecting and annotating a dataset.

Training a YOLOv8-Seg model on Google Colab with GPU.

Evaluating model performance and saving best weights.

Building a Streamlit-based web application for video uploads, inference, and annotated result downloads.

Deploying the app on Streamlit Cloud for easy access.

📂 Repository Structure
vehicle-pedestrian-tracking/
│── dataset/             # Train, Val, Test images + labels (YOLO format)
│── notebooks/           # Colab notebooks for training and evaluation
│── models/              # Trained YOLO weights (best.pt)
│── results/             # Evaluation outputs, graphs, sample runs
│── outputs/             # Inference results (annotated videos, JSONs) [ignored in git]
│── uploads/             # Uploaded videos for processing [ignored in git]
│── src/                 # Custom helper scripts
│── archive_old_webapp/  # Archived initial app attempts
│── app.py               # Streamlit web application
│── tracker.py           # Video tracking + YOLO + ByteTrack logic
│── requirements.txt     # Python dependencies
│── .gitignore           # Ignored files/folders (outputs, uploads, cache)
│── README.md            # Project documentation

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/ANIRUDH-SHARMA-25/vehicle-pedestrian-tracking.git
cd vehicle-pedestrian-tracking

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add Trained Model

Place your trained YOLO weights file (best.pt) inside the models/ folder:

models/best.pt

4️⃣ Run Locally
streamlit run app.py


Upload a video (mp4/avi/mov).

The app will run YOLOv8-Seg + ByteTrack.

Outputs:

results.json → frame-wise tracked objects.

annotated_video.mp4 → video with bounding boxes + IDs.

Both can be downloaded directly from the interface.

✅ Current Progress

Dataset collected & annotated (2 classes: vehicles, pedestrians).

YOLOv8n-Seg model trained (100 epochs on Colab GPU).

Best weights saved (models/best.pt).

Streamlit web application built & tested locally.

Successfully deployed on Streamlit Cloud 🎉.

🔗 Live Demo: Streamlit App Link

📊 Next Steps

Fine-tune model with larger dataset for higher accuracy.

Optimize inference speed for larger videos.

Extend app for multi-class tracking (e.g., bikes, buses).

Add support for real-time webcam/RTSP streams.

📦 Requirements

The app requires the following main dependencies:

streamlit>=1.10
ultralytics>=8.0.0
opencv-python-headless
numpy
pillow

📝 Notes

outputs/ and uploads/ folders are git-ignored (local runtime data only).

Archived early webapp attempts are kept in archive_old_webapp/ for reference.

ByteTrack support is modular — can be swapped or extended with other trackers.

✨ Authors: Anirudh Sharma & Team
📅 2025


