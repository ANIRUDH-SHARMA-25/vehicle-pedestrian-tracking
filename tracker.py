# tracker.py
import os, cv2, json
import numpy as np
from ultralytics import YOLO

def _draw_box(frame, bbox, label, color=(0,255,0)):
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
    cv2.putText(frame, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def process_video(video_path, model_path="models/best.pt", output_dir="outputs", conf=0.25):
    """
    Runs YOLO inference on each frame and attempts to use ByteTrack for tracking.
    If ByteTrack is not available, falls back to a simple ID assignment per detection.
    Returns (results_json_path, annotated_video_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = YOLO(model_path)

    # Try to import ByteTrack
    use_bytetrack = False
    try:
        import sys
        # allow ByteTrack to be placed at repo root as 'ByteTrack' or installed in env
        sys.path.append("ByteTrack")
        from yolox.tracker.byte_tracker import BYTETracker
        from types import SimpleNamespace
        args = SimpleNamespace(track_thresh=conf, match_thresh=0.8, track_buffer=30, mot20=False)
        tracker = BYTETracker(args, frame_rate=30)
        use_bytetrack = True
    except Exception as e:
        # ByteTrack not available; fall back
        tracker = None
        print("ByteTrack not available — falling back to simple tracker. To enable ByteTrack, follow README instructions.")
        # continue with simple tracker

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = os.path.join(output_dir, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))

    results = []
    frame_idx = 0
    next_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # run YOLO inference on the frame
        res = model.predict(frame, conf=conf, verbose=False)[0]
        # handle empty boxes safe
        if len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes = np.empty((0,4))
            scores = np.empty((0,))
            classes = np.empty((0,), dtype=int)

        if use_bytetrack and tracker is not None:
            # ByteTrack expects detections as [x1,y1,x2,y2,score]
            dets = np.concatenate([boxes, scores[:, None]], axis=1) if boxes.shape[0] > 0 else np.empty((0,5))
            online_targets = tracker.update(dets, classes, (H, W), (H, W))
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                cls_id = int(getattr(t, "cls_id", -1))
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                bbox = [x1, y1, x2, y2]
                label = model.names[cls_id] if (cls_id >= 0 and cls_id < len(model.names)) else str(cls_id)
                score = float(getattr(t, "score", 1.0))
                results.append({
                    "frame": frame_idx,
                    "track_id": int(tid),
                    "class_id": cls_id,
                    "class": label,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": score
                })
                _draw_box(frame, bbox, f"{label}-{tid}")
        else:
            # fallback: assign a new ID for every detection (not a real tracker,
            # but useful to demonstrate the app if ByteTrack isn't installed)
            for i, box in enumerate(boxes):
                next_id += 1
                x1,y1,x2,y2 = box.tolist()
                cls_id = int(classes[i]) if i < len(classes) else -1
                label = model.names[cls_id] if (cls_id >= 0 and cls_id < len(model.names)) else str(cls_id)
                score = float(scores[i]) if i < len(scores) else 1.0
                results.append({
                    "frame": frame_idx,
                    "track_id": int(next_id),
                    "class_id": cls_id,
                    "class": label,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": score
                })
                _draw_box(frame, [x1,y1,x2,y2], f"{label}-{next_id}")

        writer.write(frame)

    cap.release()
    writer.release()

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return json_path, out_video
