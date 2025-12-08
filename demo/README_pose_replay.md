# Pose Replay & Visualization – Mini README

## 1. What the topics contain

### `/human_pose_status` (diagnostics as JSON String)

Each message looks like:

data: "{
  \"status\": \"ok\",
  \"confidence\": 1.0,
  \"translation_source\": \"depth\",
  \"note\": \"depth_ok z=1.835m at (u=242, v=339) from depth_001237.png\",
  \"frame_id\": \"camera_color_optical_frame\",
  \"age_ms\": 0,
  \"x\": -0.8291796649801989,
  \"y\": -0.05614948007388686,
  \"z\": 1.835
}"

Interpretation:
- `status`: state for this frame:
  - `"ok"` → valid pose + metric translation.
  - `"stale"` → reusing last known pose (still within timeout).
  - `"no_person"`, `"no_depth_image"`, etc. → error modes.
- `confidence`: orientation confidence (0–1).
- `translation_source`: where metric position came from (`"depth"` right now).
- `note`: extra info like depth range and which depth image was used:
  - e.g. `"depth_ok z=1.835m at (u=242, v=339) from depth_001237.png"`.
- `frame_id`: typically `"camera_color_optical_frame"`.
- `age_ms`:
  - 0 for fresh pose.
  - >0 if reusing a last-known pose (stale) within timeout.
- `x, y, z`: torso position in **meters**, in camera frame.

Example meaning:
- Person is ~1.83 m in front of the camera, slightly left and slightly above the optical center.
- Orientation was confident and translation is based on depth.

---

### `/human_pose` (geometry_msgs/PoseStamped)

Example:

header:
  frame_id: "camera_color_optical_frame"
pose:
  position:
    x: -0.2354
    y: 0.0843
    z: 0.83
  orientation:
    x: -0.0109
    y: 0.2601
    z: 0.0122
    w: 0.9654

Interpretation:
- `position` = (x, y, z) in meters, in `camera_color_optical_frame`:
  - z ≈ distance from camera (~0.83 m here).
  - x, y give left/right and up/down offsets.
- `orientation` = quaternion (x, y, z, w) in camera frame:
  - represents torso orientation (heading + tilt).
  - values here are small → person is almost aligned to camera forward axis with slight tilt.

These match what the pipeline is expected to publish.

---

## 2. How `rostopic echo` behaves with replay_pose

- `replay_pose.py --loop` replays the log in a **continuous loop**.
- `rostopic echo /human_pose` (without `-n`) prints **every message as it’s published**.
- `rostopic echo /human_pose -n 1` waits for the **next** message and prints exactly one.

Important:
- You’re not seeing “the first frame” of the log, you’re sampling **wherever the loop happens to be** at the moment you run the command.
- Same logic for `/human_pose_status` and `/human_pose_marker`.

---

## 3. What is in `pose_stamped.log` / `pose_stamped_all.log`?

Each **data line** (ignoring `#` comments) has this format:

frame_index | has_person | t_ms | frame_id | conf | yaw_deg | qw qx qy qz | status | pos_xyz_m | translation_source | note

Example:

000143 | 1 | 0.00 | camera_color_optical_frame | 1.000000 | 6.129558 | 0.99524057 -0.0038528661 0.096886 0.009714718 | OK | [-0.875608370462858, -0.06234489680958154, 1.8900000000000001] | depth | depth_ok z=1.890m at (u=232, v=337) from depth_000143.png

Field meanings:
- `frame_index`: integer frame index in the sequence.
- `has_person`: 1 if a person was detected, 0 otherwise.
- `t_ms`: timestamp in milliseconds (relative, here often 0 for image mode).
- `frame_id`: usually `"camera_color_optical_frame"`.
- `conf`: confidence in orientation estimate (float, or `NA`).
- `yaw_deg`: yaw angle in degrees (or `NA`) derived from the quaternion.
- `qw qx qy qz`: torso orientation as quaternion (w, x, y, z).
- `status`:
  - `"OK"` → good pose and translation.
  - `"SKIP:no_person"`, `"SKIP:no_depth"`, `"SKIP:low_confidence(...)"`, etc.
- `pos_xyz_m`: `[x, y, z]` in meters in camera frame, or `NA` if unavailable.
- `translation_source`: e.g. `"depth"`, `"none"`, etc.
- `note`: human-readable info (e.g. which depth pixel / depth PNG was used).

`pose_stamped_all.log` is just **many such blocks concatenated**, possibly with helpful comments like:

# seq=color_000143
000143 | ...

Those `# seq=...` lines are treated as comments and are ignored by the parser.

---

## 4. What does `replay_pose.py` do?

`replay_pose.py` reads a log (e.g. `pose_stamped_all.log`) and acts like a ROS node that:

- For each `frame_index`:
  - picks the **best line** (based on `status` ranking – e.g. prefers `OK` over SKIP).
- Publishes:
  - `/human_pose` (geometry_msgs/PoseStamped) – position + quaternion.
  - `/human_pose_status` (std_msgs/String) – JSON diagnostics.
- Optionally:
  - A TF transform `camera_color_optical_frame → human_base` on `/tf`.
  - A RViz Marker (green arrow) on `/human_pose_marker`.

This means:
- Your colleague does **not** need PoseFormer or the original images to visualize.
- She only needs:
  - `replay_pose.py`
  - the combined log
  - ROS + RViz

---

## 5. How to run and verify (for colleague)

### 5.1 Start ROS core

source /opt/ros/noetic/setup.bash   # or your ROS distro
roscore

### 5.2 Basic replay (PoseStamped + diagnostics only)

In a new terminal:

cd /path/to/project
python demo/replay_pose.py \
    --log demo/output/pose_stamped_all.log \
    --hz 10 \
    --loop

Check topics:

rostopic list
rostopic hz /human_pose
rostopic echo /human_pose -n 3
rostopic echo /human_pose_status -n 3

What you see:
- `/human_pose` → sequence of PoseStamped with x,y,z + quaternion.
- `/human_pose_status` → JSON stats (status, confidence, translation source, note, etc.).

Remember:
- Because of `--loop`, you’re sampling the **live loop**, not necessarily the first frame.

### 5.3 Replay with TF + RViz Marker (for visualization)

python demo/replay_pose.py \
    --log demo/output/pose_stamped_all.log \
    --hz 10 \
    --loop \
    --publish-tf --tf-child-frame human_base \
    --publish-marker --marker-topic /human_pose_marker --marker-scale 0.2

Sanity checks in terminal:

# Marker exists
rostopic echo /human_pose_marker -n 1

# TF packets (raw)
rostopic echo /tf -n 1

# Human frame relative to camera, nicely formatted:
rosrun tf tf_echo camera_color_optical_frame human_base

You should see something like:

Translation: [x, y, z]
Rotation: in Quaternion [x, y, z, w]
          in RPY (degree) [roll, pitch, yaw]

---

### 5.4 Visualizing in RViz

1. Run RViz:

   rviz

2. In RViz:
   - Set **Fixed Frame** = `camera_color_optical_frame`.
   - Add a **TF** display:
     - You should see a frame called `human_base` under the camera frame.
   - Add a **Marker** display:
     - Topic: `/human_pose_marker`
     - You’ll see a **green arrow** where the torso is in 3D.
   - (Optional) Add a **Pose** display:
     - Topic: `/human_pose`

With `--loop` enabled, the arrow (and/or pose) will keep updating as the log cycles through all frames.

---

## 6. Notes / Gotchas

- Use `--loop` during inspection, or the replayer will publish once and exit.
- `rostopic echo /human_pose -n 1` gives you the **next** message at the moment you call it, not necessarily the start of the log.
- TF and Marker are **not** embedded inside `/human_pose`. They live on:
  - TF: `/tf`
  - Marker: `/human_pose_marker`
- Your combined log format (with `# seq=...` comments and the `frame_index | ...` lines) is compatible with `replay_pose.py` as-is; no preprocessing is required.

