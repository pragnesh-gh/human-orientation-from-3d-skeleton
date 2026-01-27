# PoseFormerV2 RGB-D Pose and Orientation Pipeline

## Quickstart (the fast way)

This repo is **meant for RGB-D inputs** (RGB + aligned depth).  
You *can* run it with **RGB-only** as well (you’ll still get 2D/3D pose + orientation, but metric translation will be missing).

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Make sure these pretrained weights exist in the repo:
```text
checkpoint/27_243_45.2.bin
demo/lib/checkpoint/pose_hrnet_w48_384x288.pth
demo/lib/checkpoint/yolov3.weights
```

3) Run PoseFormerV2 + orientation on a video (produces visualizations + log)
```bash
python demo/vis.py --video output_video.mp4 --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0 --log-pose-stamped
```

4) (Optional) Replay the saved log in ROS (Linux ROS machine)
```bash
python demo/replay_pose.py --log demo/output/<run_name>/pose_stamped.log --hz 10 --loop
```

---

## 1. Overview

This repository contains a **human pose estimation pipeline** built on top of **PoseFormerV2**, extended to work with **RGB-D data** and to estimate the **heading (orientation) of a person**.

What you get:
- 2D pose (COCO-17) using **YOLOv3 + SORT + HRNet**
- 3D pose (H36M-17) using **PoseFormerV2**
- Torso-based **orientation / heading** as a quaternion + basis vectors
- Optional **metric translation** (X,Y,Z in meters) using depth + intrinsics
- Visualizations + a reusable log (`pose_stamped.log`) that can be replayed in ROS

A practical note about **videos**: a video is treated as a **sequence of frames**. The pipeline preserves frame indexing and tries hard not to break time alignment.

---

## 2. Repository structure (what is where?)

```text
PoseFormerV2_latest/
│
├── demo/              # End-to-end demo pipeline and ROS tools
├── common/            # Shared utilities (model, camera, orientation, math)
├── checkpoint/        # PoseFormerV2 checkpoint (required)
├── requirements.txt   # Python dependencies
└── run_poseformer.py  # Training / evaluation script (not used in demo workflow)
```

### 2.1 `demo/` — end-to-end usage and integration

#### `demo/vis.py` — main entry point (run this first)

This script runs the full pipeline:
- 2D pose extraction
- COCO→H36M conversion
- PoseFormerV2 3D pose
- Orientation estimation (optional overlay / saving)
- Optional depth-based translation (RGB-D)
- Writes outputs (visualizations + logs)

#### `demo/replay_pose.py` — ROS replay tool (run this later)

This script reads `pose_stamped.log` produced by `vis.py` and republishes it in ROS:
- `/human_pose` (PoseStamped)
- `/human_pose_status` (diagnostics)
- optional TF + RViz marker publishing

#### `demo/lib/` — pipeline helpers (used by `vis.py`)

- `demo/lib/hrnet/gen_kpts.py`: YOLO + SORT + HRNet 2D keypoints (COCO-17).  
  Output shapes: `(M, T, 17, 2)` keypoints and `(M, T, 17)` scores.
- `demo/lib/preprocess.py`: COCO-17 → H36M-17 conversion + repair logic.
- `demo/lib/yolov3/human_detector.py`: loads YOLO and returns human boxes + scores.
- `demo/lib/yolov3/darknet.py`: YOLO network and `.weights` loader.

### 2.2 `common/` — shared core logic

- `common/model_poseformer.py`: PoseFormerV2 model definition (2D→3D).
- `common/camera.py`: screen-coordinate normalization, camera/world transforms, projection helpers.
- `common/orientation.py`: torso frame + orientation quaternion + confidence + smoothing.
- `common/quaternion.py`: quaternion math utilities (SLERP, rotmat conversions).

---

## 3. How the pipeline works (conceptual view)

```text
RGB image / video frames
        ↓
Person detection (YOLOv3)
        ↓
Tracking (SORT)
        ↓
2D pose estimation (HRNet, COCO-17 joints + per-joint confidence)
        ↓
COCO-17 → H36M-17 conversion (preprocess.py)
        ↓
2D → 3D pose lifting (PoseFormerV2)
        ↓
Torso orientation estimation (orientation.py)
        ↓
Optional depth-based translation (RGB-D)
        ↓
Visualizations + pose_stamped.log (+ optional CSV)
```

---

## 4. Setup

### 4.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 4.2 Required model files (must exist)

PoseFormerV2 checkpoint (**hard-coded by vis.py**):
```text
checkpoint/27_243_45.2.bin
```

HRNet weights:
```text
demo/lib/checkpoint/pose_hrnet_w48_384x288.pth
```

YOLOv3 weights:
```text
demo/lib/checkpoint/yolov3.weights
```

---

## 5. How to run (correct commands)

`vis.py` supports three main input modes:
- `--image` (single image)
- `--image-dir` (folder of images, e.g., a video extracted to frames)
- `--video` (video file)

### 5.1 Video (RGB-only)

```bash
python demo/vis.py --video output_video.mp4 --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0
```

Same, but also write the PoseStamped log:
```bash
python demo/vis.py --video output_video.mp4 --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0 --log-pose-stamped
```

### 5.2 Single image (RGB-only)

```bash
python demo/vis.py --image frame0141.jpg --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0
```

If you want the log to go to a specific location:
```bash
python demo/vis.py --image frame0141.jpg --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0 --ros-log-file ./demo/output/frame0141/pose_stamped.log
```

### 5.3 Single image (RGB-D translation)

```bash
python demo/vis.py --image new_test/color_0480.png --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0 --log-pose-stamped \
  --translation-source depth --depth-path new_test/depth/depth_0480.png --depth-scale 0.001 \
  --fx 911.47 --fy 911.56 --cx 654.27 --cy 366.90
```

Notes:
- `--depth-scale 0.001` is typical when depth PNG stores **millimeters** and you want **meters**.

### 5.4 Folder of images (RGB-D, and save orientation CSV)

```bash
python demo/vis.py --image-dir new_test/color --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0 \
  --translation-source depth --depth-dir new_test/depth --depth-scale 0.001 \
  --fx 911.47 --fy 911.56 --cx 654.27 --cy 366.90 \
  --log-pose-stamped --orientation-save orientation.csv
```

### 5.5 RGB-D for video mode (important note)

For video inputs, use **directory + pattern** instead of a single depth path:

- Use `--depth-dir` and `--depth-pattern` (not `--depth-path`) with `--video`.

(Exact pattern depends on how your depth PNGs are named.)

---

## 6. Outputs (what you should expect)

`vis.py` produces an output folder under:

```text
demo/output/<run_name>/
```

Typical contents:
- `pose2D/` : 2D overlays
- `pose3D/` : 3D overlays
- `pose/`   : combined view (if enabled)
- `input_2D/keypoints.npz` : saved 2D keypoints input to PoseFormer
- `pose_stamped.log` : pose + orientation (+ translation if depth) per frame (if enabled)

---

## 7. ROS replay (how to run & verify)

### 7.1 Start ROS core
```bash
source /opt/ros/noetic/setup.bash
roscore
```

### 7.2 Replay PoseStamped + diagnostics (loop so you can inspect)
```bash
python demo/replay_pose.py --log demo/color_0141/pose_stamped.log --hz 10 --loop
```

Inspect in terminal:
```bash
rostopic list
rostopic hz /human_pose
rostopic echo /human_pose -n 3
rostopic echo /human_pose_status -n 3
```

### 7.3 Add TF + RViz Marker (optional visualization outputs)
```bash
python demo/replay_pose.py --log demo/color_0141/pose_stamped.log --hz 10 --loop \
  --publish-tf --tf-child-frame human_base \
  --publish-marker --marker-topic /human_pose_marker --marker-scale 0.2
```

TF/Marker are NOT part of `/human_pose`:
```bash
rostopic echo /human_pose_marker -n 1
rostopic echo /tf -n 1
rosrun tf tf_echo camera_color_optical_frame human_base
```

### 7.4 RViz (GUI)
```bash
rviz
```

Suggested settings:
- Fixed Frame: `camera_color_optical_frame`
- Add “TF” display (to see `human_base` under the camera frame)
- Add “Marker” display, Topic: `/human_pose_marker`
- Optionally add “Pose” display, Topic: `/human_pose`

Notes:
- Use `--loop` during inspection; otherwise the replayer publishes once and exits.
- If you don’t pass `--publish-tf/--publish-marker`, replay is still valid (just fewer visual aids).

---

## 8. Notes and design decisions

- **Timeline integrity (no dropped frames)**  
  The 2D frontend does not drop frames. If no detection/tracking is available, it writes zeros for that frame to preserve alignment.

- **2D normalization** attaches to PoseFormer input  
  PoseFormer expects normalized screen coordinates, not raw pixels:
  - X maps to `[-1, 1]`
  - Y is scaled by `h / w` to preserve aspect ratio

- **Quaternion convention**  
  Internal quaternions are **wxyz**. ROS expects **xyzw**, so reordering is done before publishing.

- **Orientation definition**  
  Orientation is torso-based (hips + shoulders), not head-based.

- **RGB-D translation**  
  Metric translation needs aligned depth and intrinsics. RGB-only runs are supported but won’t give metric translation.

---

## 9. Troubleshooting

- **Model loading error like “list index out of range”**  
  Usually means PoseFormer checkpoint is missing. Ensure:
  ```text
  checkpoint/27_243_45.2.bin
  ```

- **No translation / Z is zero**  
  Usually means depth is missing, not aligned, or depth-scale/intrinsics are wrong.

---

