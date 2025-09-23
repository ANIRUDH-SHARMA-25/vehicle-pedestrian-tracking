Vehicle and Pedestrian Detection & Tracking
📌 Project Overview

This project implements a YOLOv8 segmentation model for detecting and tracking vehicles and pedestrians.
The workflow includes:

Collecting and preparing datasets (Unsplash, Pexels, OpenImages).

Annotating data using Labellerr.

Training a YOLOv8-seg model on Google Colab with GPU.

Evaluating performance on test images.

Preparing results and analysis for submission.

.

📂 Repository Structure
project_repo/
│── dataset/          # Train, Val, Test images + labels (YOLO format)
│── notebooks/        # Colab notebooks for training and evaluation
│── models/           # Trained YOLO weights (.pt files)
│── results/          # Evaluation outputs, graphs, and demo runs
│── src/              # Custom helper scripts (if needed)
│── README.md         # Project documentation
│── requirements.txt  # Python dependencies

⚙️ Setup Instructions

Clone the Repository
git clone <repo-link>
cd project_repo

install Dependencies
pip install -r requirements.txt

Training the Model
Training was done using Google Colab with Tesla T4 GPU
yolo segment train model=yolov8n-seg.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=8




✅ Current Progress

 Dataset collected (183 images → split into train/val/test).

 Annotation completed in Labellerr (2 classes: vehicles, pedestrians).

 Labels exported in YOLO format.

 Dataset prepared in dataset/.

 Training started (YOLOv8n-seg, 100 epochs).

 Evaluation on test set (next).

 Results visualization.

 Final report preparation.


📊 Next Steps

Finish training (100 epochs).

Evaluate model performance on test set.

Save best weights (best.pt) to models/.

Generate precision-recall curves and segmentation outputs.

Document results in results/ and update README.
