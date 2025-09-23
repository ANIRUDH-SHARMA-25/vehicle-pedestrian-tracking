# app.py — Streamlit Video Tracking Demo (YOLOv8-seg + ByteTrack)
import streamlit as st
import tempfile, os, json, time
from ultralytics import YOLO
from pathlib import Path
import shutil
from tqdm import tqdm

st.set_page_config(page_title="YOLOv8 + ByteTrack Demo", layout="wide")

st.title("Video Tracking Demo — YOLOv8-seg + ByteTrack")
st.markdown("Upload a video, the app will run detection + ByteTrack and produce an annotated video and `results.json`.")

# --- Sidebar: settings
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model path (local or drive)", "/content/drive/MyDrive/project_name/models/best.pt")
tracker_cfg = st.sidebar.text_input("Tracker config (ultralytics tracker)", "bytetrack.yaml")
confidence = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.45)
device = st.sidebar.text_input("Device (e.g. cpu or cuda:0)", "cuda:0")  # change to "cpu" if no GPU

# Upload widget
uploaded = st.file_uploader("Upload a video (.mp4, .mov, .avi). Max ~100MB for demo.", type=["mp4","mov","avi"])
run_btn = st.button("Run Tracking")

# Utility: write results JSON
def save_results_json(out_path, items):
    with open(out_path, "w") as f:
        json.dump(items, f, indent=2)

# Main run
if run_btn:
    if not uploaded:
        st.warning("Please upload a video file first.")
    else:
        # prepare temp files
        tmpdir = tempfile.mkdtemp()
        input_video_path = os.path.join(tmpdir, uploaded.name)
        with open(input_video_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.info(f"Saved uploaded file to `{input_video_path}`. Loading model...")

        # load model
        try:
            model = YOLO(model_path)
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Failed to load model from `{model_path}`: {e}")
            raise

        # set output folders
        timestamp = int(time.time())
        output_parent = Path(tmpdir) / f"output_{timestamp}"
        output_parent.mkdir(parents=True, exist_ok=True)
        project_output = str(output_parent)

        st.info("Running tracker — this can take a while depending on GPU/CPU...")
        progress_text = st.empty()
        bar = st.progress(0)

        # Run tracking using ultralytics track (best-effort)
        try:
            results = model.track(
                source=input_video_path,
                tracker=tracker_cfg,
                save=True,                 # ask ultralytics to save annotated output
                project=project_output,    # where ultralytics will write results
                name="track_run",
                conf=confidence,
                device=device,
            )
        except Exception as e:
            st.error(f"Tracker run failed: {e}")
            st.info("If tracker config not found, try 'botsort.yaml' or update ultralytics. See instructions.")
            raise

        # results is an iterable of per-frame Results
        st.success("Tracking run completed. Extracting track info...")

        # Build result list
        results_list = []
        frame_idx = 0
        categories = {0: "pedestrians", 1: "vehicles"}  # update if your class mapping differs

        for r in results:  # results yields per-frame Results
            # try to access boxes in a robust way
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                frame_idx += 1
                continue

            # common representation: boxes.data (torch tensor) columns:
            # [x1, y1, x2, y2, score, class, track_id?]
            data = None
            try:
                data = boxes.data.cpu().numpy()  # numpy array (N, >=6)
            except Exception:
                # fallback: try xyxy + cls + conf arrays
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    # build rows
                    rows = []
                    for i in range(len(cls)):
                        x1,y1,x2,y2 = xyxy[i].tolist()
                        rows.append([x1,y1,x2,y2, confs[i], int(cls[i])])
                    data = np.array(rows)
                except Exception:
                    data = None

            if data is None:
                frame_idx += 1
                continue

            for det in data:
                # det could be length 6 or 7 (if track id present)
                x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                score = float(det[4])
                cls_id = int(det[5])
                track_id = int(det[6]) if det.shape[0] > 6 else None

                # bbox as [x,y,w,h]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                item = {
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": categories.get(cls_id, str(cls_id)),
                    "bbox": [round(v, 2) for v in bbox],
                    "score": round(score, 3)
                }
                results_list.append(item)
            frame_idx += 1

        # locate annotated video saved by ultralytics
        saved_dir = Path(project_output) / "track_run"
        annotated_video = None
        if saved_dir.exists():
            # find video file (mp4/avi)
            for ext in ("mp4","avi","mkv"):
                cand = list(saved_dir.glob(f"**/*.{ext}"))
                if cand:
                    annotated_video = str(cand[0])
                    break

        # Save results.json
        results_json_path = str(Path(saved_dir) / "results.json") if annotated_video else str(output_parent / "results.json")
        save_results_json(results_json_path, results_list)

        st.success(f"Saved results.json ({len(results_list)} detections) at:\n`{results_json_path}`")
        if annotated_video:
            st.success(f"Annotated video saved at `{annotated_video}`")
            st.video(annotated_video)
            st.markdown(f"[Download annotated video]({annotated_video})")
        else:
            st.warning("Annotated video not found in expected output folder. Check ultralytics tracker saving behavior.")

        st.download_button("Download results.json", data=json.dumps(results_list, indent=2), file_name="results.json")
        st.write("Temporary output folder:", str(output_parent))
        st.info("When done, delete the temporary folder to free space.")
