#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, time, json
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

# >>> NEW: TF + Marker imports (safe on ROS machine) <<<
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
import tf2_ros

LINE_RE = re.compile(
    r'^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(\S+)\s*\|\s*([NA0-9.]+)\s*\|\s*([NA0-9.\-]+)\s*\|\s*([0-9eE.\-\s]+)\s*\|\s*([A-Za-z:_()0-9]+)\s*\|\s*(\[.*?\]|NA)\s*\|\s*(\w+)\s*\|\s*(.*)\s*$'
)
# fields: frame_index | has_person | t_ms | frame_id | conf | yaw_deg | qw qx qy qz | status | pos_xyz_m | translation_source | note

def parse_line(line):
    s = line.strip()
    if not s or s.startswith('#'):
        return None
    m = LINE_RE.match(s)
    if not m:
        return None
    frame_idx   = int(m.group(1))
    has_person  = bool(int(m.group(2)))
    t_ms        = float(m.group(3))
    frame_id    = m.group(4)

    conf        = None if m.group(5) == 'NA' else float(m.group(5))
    yaw_deg     = None if m.group(6) == 'NA' else float(m.group(6))

    qwqxqyqz    = [float(x) for x in m.group(7).strip().split()]
    status      = m.group(8)

    pos_xyz_raw = m.group(9)
    if pos_xyz_raw == 'NA':
        pos_xyz = None
    else:
        nums = [float(x) for x in re.findall(r'[-+]?[0-9]*\.?[0-9]+', pos_xyz_raw)]
        pos_xyz = nums[:3] if len(nums) >= 3 else None

    trans_src   = m.group(10)
    note        = m.group(11)

    return {
        "frame_idx": frame_idx,
        "has_person": has_person,
        "t_ms": t_ms,
        "frame_id": frame_id,
        "conf": conf,
        "yaw_deg": yaw_deg,
        "quat": qwqxqyqz,        # [w, x, y, z]
        "status": status,        # "OK", "SKIP:no_depth", "SKIP:no_person", ...
        "pos": pos_xyz,          # [x, y, z] (meters) or None
        "translation_source": trans_src,
        "note": note,
    }

# ---- Status ranking: higher is better (keep the single best line per frame) ----
def status_rank(s: str) -> int:
    if s == "OK":
        return 5
    if s.startswith("SKIP:low_confidence("):
        return 3
    if s == "SKIP:orientation_disabled":
        return 2
    if s == "SKIP:no_depth":
        return 1
    if s == "SKIP:no_person":
        return 0
    return -1  # unknown/lowest

def better(a, b):
    """Return the 'better' row by status_rank; break ties by choosing the later one."""
    if a is None:
        return b
    if b is None:
        return a
    ra, rb = status_rank(a["status"]), status_rank(b["status"])
    if rb > ra:
        return b
    if rb < ra:
        return a
    # tie: prefer the one that has pos (if only one has it), else the later in the file (b)
    if (a["pos"] is None) and (b["pos"] is not None):
        return b
    return b

def publish_pose(pose_pub, frame_id, pos, quat_wxyz):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    # position
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
    # orientation: ROS uses x,y,z,w
    msg.pose.orientation.x = quat_wxyz[1]
    msg.pose.orientation.y = quat_wxyz[2]
    msg.pose.orientation.z = quat_wxyz[3]
    msg.pose.orientation.w = quat_wxyz[0]
    pose_pub.publish(msg)

# >>> NEW: RViz Marker + TF helpers (minimal) <<<
def make_marker(frame_id, pos, quat_wxyz, ns="human_pose", mid=1, scale=0.15):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.scale.x = scale * 1.5   # shaft length
    m.scale.y = scale * 0.2   # shaft diameter
    m.scale.z = scale * 0.2
    m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.8, 0.2, 0.9
    m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
    # quat_wxyz (w,x,y,z) -> ROS order x,y,z,w
    m.pose.orientation.x = quat_wxyz[1]
    m.pose.orientation.y = quat_wxyz[2]
    m.pose.orientation.z = quat_wxyz[3]
    m.pose.orientation.w = quat_wxyz[0]
    return m

def send_tf(tf_broadcaster, parent_frame, child_frame, pos, quat_wxyz):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.transform.translation.x = pos[0]
    t.transform.translation.y = pos[1]
    t.transform.translation.z = pos[2]
    t.transform.rotation.x = quat_wxyz[1]
    t.transform.rotation.y = quat_wxyz[2]
    t.transform.rotation.z = quat_wxyz[3]
    t.transform.rotation.w = quat_wxyz[0]
    tf_broadcaster.sendTransform(t)

def commit_frame(best_row, lkv, pose_pub, diag_pub, timeout_ms,
                 marker_pub=None, tf_broadcaster=None, tf_child_frame="human_base", marker_scale=0.15):
    """
    Commit a single, best line for the current frame:
      - If row is OK with valid pos, publish fresh and update LKV.
      - Else, try to republish last-known if still fresh (stale).
      - Emit a diagnostics String JSON either way.
      - >>> NEW: Publish Marker/TF whenever we publish a PoseStamped (fresh OR stale within timeout).
    """
    if best_row is None:
        return

    now_ms   = int(time.time() * 1000)
    frame_id = best_row["frame_id"]
    status   = best_row["status"]
    conf     = best_row["conf"] if best_row["conf"] is not None else 1.0
    trans    = best_row["translation_source"]
    note     = best_row["note"]

    diag = {
        "status": None,
        "confidence": round(conf, 3),
        "translation_source": trans,
        "note": note,
        "frame_id": frame_id,
        "age_ms": 0
    }

    # Fresh OK with translation → publish and store
    if status == "OK" and best_row["pos"] is not None:
        pos  = tuple(best_row["pos"])
        quat = tuple(best_row["quat"])
        publish_pose(pose_pub, frame_id, pos, quat)
        # NEW: Marker/TF on fresh publish
        if marker_pub is not None:
            marker_pub.publish(make_marker(frame_id, pos, quat, scale=marker_scale))
        if tf_broadcaster is not None:
            send_tf(tf_broadcaster, frame_id, tf_child_frame, pos, quat)

        lkv["pose"] = (frame_id, pos, quat)
        lkv["t_wall_ms"] = now_ms
        diag["status"] = "ok"
        diag["x"], diag["y"], diag["z"] = pos
    else:
        # Not OK: try stale republish if we have a recent last-known pose
        if lkv["pose"] is not None:
            age = now_ms - (lkv.get("t_wall_ms") or now_ms)
            diag["age_ms"] = int(age)
            if age <= timeout_ms:
                frame_id_l, pos_l, quat_l = lkv["pose"]
                publish_pose(pose_pub, frame_id_l, pos_l, quat_l)
                # NEW: Marker/TF for stale publish (only when within timeout)
                if marker_pub is not None:
                    marker_pub.publish(make_marker(frame_id_l, pos_l, quat_l, scale=marker_scale))
                if tf_broadcaster is not None:
                    send_tf(tf_broadcaster, frame_id_l, tf_child_frame, pos_l, quat_l)

                diag["status"] = "stale"
                diag["x"], diag["y"], diag["z"] = pos_l
            else:
                diag["status"] = "no_person" if status.startswith("SKIP:no_person") else (
                    "no_depth_image" if status.startswith("SKIP:no_depth") else status.lower()
                )
        else:
            diag["status"] = "no_person" if status.startswith("SKIP:no_person") else (
                "no_depth_image" if status.startswith("SKIP:no_depth") else status.lower()
            )

    diag_pub.publish(String(data=json.dumps(diag)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to pose_stamped.log")
    ap.add_argument("--hz", type=float, default=10.0, help="publish rate (frames/sec)")
    ap.add_argument("--topic", default="/human_pose", help="PoseStamped topic")
    ap.add_argument("--diag-topic", default="/human_pose_status", help="diagnostics topic (String JSON)")
    ap.add_argument("--stale-timeout-ms", type=int, default=1000, help="max age to republish last-known pose")
    ap.add_argument("--loop", action="store_true", help="loop the log")

    # >>> NEW flags for TF + Marker <<<
    ap.add_argument("--publish-tf", action="store_true", help="broadcast TF frame for the human pose")
    ap.add_argument("--tf-child-frame", default="human_base", help="child TF frame name")
    ap.add_argument("--publish-marker", action="store_true", help="publish RViz Marker for the human pose")
    ap.add_argument("--marker-topic", default="/human_pose_marker", help="Marker topic")
    ap.add_argument("--marker-scale", type=float, default=0.15, help="Marker arrow size")

    args = ap.parse_args()

    rospy.init_node("pose_replayer", anonymous=True)
    pose_pub = rospy.Publisher(args.topic, PoseStamped, queue_size=10)
    diag_pub = rospy.Publisher(args.diag_topic, String, queue_size=10)

    # >>> NEW: Optional Marker/TF publishers (created only if flags set) <<<
    marker_pub = rospy.Publisher(args.marker_topic, Marker, queue_size=10) if args.publish_marker else None
    tf_broadcaster = tf2_ros.TransformBroadcaster() if args.publish_tf else None

    rate = rospy.Rate(args.hz)

    # last-known valid pose (for stale)
    lkv = {"pose": None, "t_wall_ms": None}
    timeout = args.stale_timeout_ms

    while not rospy.is_shutdown():
        current_frame = None
        best_row = None

        with open(args.log, "r", encoding="utf-8") as f:
            for line in f:
                if rospy.is_shutdown():
                    break
                row = parse_line(line)
                if row is None:
                    continue

                # If we moved to a new frame, commit the best from the previous frame
                if (current_frame is not None) and (row["frame_idx"] != current_frame):
                    commit_frame(best_row, lkv, pose_pub, diag_pub, timeout,
                                 marker_pub=marker_pub, tf_broadcaster=tf_broadcaster,
                                 tf_child_frame=args.tf_child_frame, marker_scale=args.marker_scale)
                    best_row = None
                    rate.sleep()

                current_frame = row["frame_idx"]
                best_row = better(best_row, row)

            # EOF: commit the last buffered frame
            if not rospy.is_shutdown():
                commit_frame(best_row, lkv, pose_pub, diag_pub, timeout,
                             marker_pub=marker_pub, tf_broadcaster=tf_broadcaster,
                             tf_child_frame=args.tf_child_frame, marker_scale=args.marker_scale)
                rate.sleep()

        if not args.loop or rospy.is_shutdown():
            break

if __name__ == "__main__":
    main()

# ---------------------- How to run & verify (cheatsheet) ----------------------
# 1) Start ROS core:
#    source /opt/ros/noetic/setup.bash
#    roscore
#
# 2) Replay PoseStamped + diagnostics (loop so you can inspect):
#    python demo/replay_pose.py --log demo/output/color_0141/pose_stamped.log --hz 10 --loop
#
#    Inspect in terminal:
#      rostopic list
#      rostopic hz /human_pose
#      rostopic echo /human_pose -n 3
#      rostopic echo /human_pose_status -n 3
#
# 3) Add TF + RViz Marker (optional visualization outputs):
#    python demo/replay_pose.py --log demo/output/color_0141/pose_stamped.log --hz 10 --loop \
#        --publish-tf --tf-child-frame human_base \
#        --publish-marker --marker-topic /human_pose_marker --marker-scale 0.2
#
#    TF/Marker are NOT part of /human_pose:
#      - Marker: rostopic echo /human_pose_marker -n 1      # (message exists; for RViz display)
#      - TF:     rostopic echo /tf -n 1                     # (raw TF packets; noisy)
#               rosrun tf tf_echo camera_color_optical_frame human_base
#                # prints the transform being broadcast (updates continuously)
#
# 4) RViz (GUI):
#    rviz
#      - Fixed Frame: camera_color_optical_frame
#      - Add "TF" display  (to see 'human_base' under the camera frame)
#      - Add "Marker" display, Topic: /human_pose_marker  (green arrow at the pose)
#      - (Optionally) Add "Pose" display, Topic: /human_pose
#
# Notes:
#  - Use --loop during inspection; otherwise the replayer publishes once and exits.
#  - To see a last message even after exit, make publishers latched (code change).
#  - If you don't pass --publish-tf/--publish-marker, behavior is identical to plain replay.

