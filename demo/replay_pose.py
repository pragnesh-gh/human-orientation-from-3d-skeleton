#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay PoseStamped and diagnostics from a saved pose_stamped.log.
Optionally publish TF and an RViz Marker for visualization.

- Core topics:
  /human_pose (geometry_msgs/PoseStamped)
  /human_pose_status (std_msgs/String)  # JSON diagnostics

- Optional:
  --publish-tf: broadcast TF transform (camera_color_optical_frame -> human_base)
  --publish-marker: publish a Marker (arrow) at the pose

Log format (one line per frame):
# fields: frame_index | has_person | t_ms | frame_id | conf | yaw_deg | qw qx qy qz | status | pos_xyz_m | translation_source | note
"""

import argparse, re, time, json
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import tf2_ros

# ---------------------- Parsing ----------------------

LINE_RE = re.compile(
    r'^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(\S+)\s*\|\s*([NA0-9.]+)\s*\|\s*([NA0-9.\-]+)\s*\|\s*([0-9eE.\-\s]+)\s*\|\s*([A-Z:]+)\s*\|\s*(\[.*?\]|NA)\s*\|\s*(\w+)\s*\|\s*(.*)\s*$'
)
# fields: frame_index | has_person | t_ms | frame_id | conf | yaw_deg | qw qx qy qz | status | pos_xyz_m | translation_source | note

def parse_line(line):
    if line.strip().startswith('#') or not line.strip():
        return None
    m = LINE_RE.match(line)
    if not m:
        return None
    frame_idx = int(m.group(1))
    has_person = bool(int(m.group(2)))
    t_ms = float(m.group(3))
    frame_id = m.group(4)
    conf = None if m.group(5) == 'NA' else float(m.group(5))
    yaw_deg = None if m.group(6) == 'NA' else float(m.group(6))
    qwqxqyqz = [float(x) for x in m.group(7).strip().split()]
    status = m.group(8)
    pos_xyz_raw = m.group(9)
    if pos_xyz_raw == 'NA':
        pos_xyz = None
    else:
        nums = [float(x) for x in re.findall(r'[-+]?[0-9]*\.?[0-9]+', pos_xyz_raw)]
        pos_xyz = nums[:3] if len(nums) >= 3 else None
    trans_src = m.group(10)
    note = m.group(11)
    return {
        "frame_idx": frame_idx, "has_person": has_person, "t_ms": t_ms, "frame_id": frame_id,
        "conf": conf, "yaw_deg": yaw_deg, "quat": qwqxqyqz, "status": status,
        "pos": pos_xyz, "translation_source": trans_src, "note": note
    }

# ---------------------- Helpers ----------------------

def publish_pose(pose_pub, frame_id, pos, quat_wxyz):
    """
    Publish geometry_msgs/PoseStamped (ROS expects orientation as x,y,z,w).
    quat_wxyz: (w, x, y, z)
    """
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
    msg.pose.orientation.x = quat_wxyz[1]
    msg.pose.orientation.y = quat_wxyz[2]
    msg.pose.orientation.z = quat_wxyz[3]
    msg.pose.orientation.w = quat_wxyz[0]
    pose_pub.publish(msg)

def publish_diag(diag_pub, diag_dict):
    diag_pub.publish(String(data=json.dumps(diag_dict)))

def make_marker(frame_id, pos, quat_wxyz, ns="human_pose", mid=1, scale=0.1):
    """
    Build an RViz Marker (arrow) at the given pose.
    """
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.scale.x = scale * 1.5  # shaft length
    m.scale.y = scale * 0.2  # shaft diameter
    m.scale.z = scale * 0.2
    m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.8, 0.2, 0.9
    m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
    # ROS uses x,y,z,w
    m.pose.orientation.x = quat_wxyz[1]
    m.pose.orientation.y = quat_wxyz[2]
    m.pose.orientation.z = quat_wxyz[3]
    m.pose.orientation.w = quat_wxyz[0]
    return m

def send_tf(tf_broadcaster, parent_frame, child_frame, pos, quat_wxyz):
    """
    Broadcast a TF transform: parent_frame -> child_frame at pose.
    """
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

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to pose_stamped.log")
    ap.add_argument("--hz", type=float, default=10.0, help="publish rate")
    ap.add_argument("--topic", default="/human_pose", help="PoseStamped topic")
    ap.add_argument("--diag-topic", default="/human_pose_status", help="diagnostics topic (String JSON)")
    ap.add_argument("--stale-timeout-ms", type=int, default=1000)
    ap.add_argument("--loop", action="store_true", help="loop the log")

    # New RViz + TF flags
    ap.add_argument("--publish-tf", action="store_true", help="broadcast TF frame for the human pose")
    ap.add_argument("--tf-child-frame", default="human_base", help="child TF frame name")
    ap.add_argument("--publish-marker", action="store_true", help="publish RViz Marker for the human pose")
    ap.add_argument("--marker-topic", default="/human_pose_marker", help="Marker topic")
    ap.add_argument("--marker-scale", type=float, default=0.15, help="Marker arrow size")

    args = ap.parse_args()

    rospy.init_node("pose_replayer", anonymous=True)
    pose_pub = rospy.Publisher(args.topic, PoseStamped, queue_size=10)
    diag_pub = rospy.Publisher(args.diag_topic, String, queue_size=10)
    marker_pub = rospy.Publisher(args.marker_topic, Marker, queue_size=10) if args.publish_marker else None
    tf_broadcaster = tf2_ros.TransformBroadcaster() if args.publish_tf else None

    rate = rospy.Rate(args.hz)

    # last-known valid
    lkv = {"pose": None, "t_wall_ms": None}  # pose: (frame_id, pos(xyz), quat(wxyz))
    timeout = args.stale_timeout_ms

    while not rospy.is_shutdown():
        with open(args.log, "r", encoding="utf-8") as f:
            for line in f:
                if rospy.is_shutdown():
                    break

                row = parse_line(line)
                if not row:
                    continue

                frame_id = row["frame_id"]
                conf = row["conf"] if row["conf"] is not None else 1.0
                trans_src = row["translation_source"]
                note = row["note"]
                now_ms = int(time.time() * 1000)

                diag = {
                    "status": None,
                    "confidence": round(conf, 3),
                    "translation_source": trans_src,
                    "note": note,
                    "frame_id": frame_id,
                    "age_ms": 0
                }

                if row["pos"] is not None:
                    # Fresh valid frame
                    pos = tuple(row["pos"])
                    quat = tuple(row["quat"])  # (w,x,y,z)

                    # Publish core pose
                    publish_pose(pose_pub, frame_id, pos, quat)
                    # Update last-known
                    lkv["pose"] = (frame_id, pos, quat)
                    lkv["t_wall_ms"] = now_ms

                    # Diagnostics
                    diag["status"] = "ok"
                    diag["x"], diag["y"], diag["z"] = pos
                    publish_diag(diag_pub, diag)

                    # Optional: Marker + TF
                    if marker_pub is not None:
                        marker_pub.publish(make_marker(frame_id, pos, quat, scale=args.marker_scale))
                    if tf_broadcaster is not None:
                        child = rospy.get_param("~tf_child_frame", args.tf_child_frame)
                        send_tf(tf_broadcaster, frame_id, child, pos, quat)

                else:
                    # Invalid frame -> republish last-known if still fresh enough
                    if lkv["pose"] is not None:
                        age = now_ms - (lkv["t_wall_ms"] or now_ms)
                        diag["age_ms"] = int(age)
                        if age <= timeout:
                            frame_id_l, pos_l, quat_l = lkv["pose"]
                            publish_pose(pose_pub, frame_id_l, pos_l, quat_l)
                            diag["status"] = "stale"
                            publish_diag(diag_pub, diag)
                            # Keep Marker/TF in sync with what we just published
                            if marker_pub is not None:
                                marker_pub.publish(make_marker(frame_id_l, pos_l, quat_l, scale=args.marker_scale))
                            if tf_broadcaster is not None:
                                child = rospy.get_param("~tf_child_frame", args.tf_child_frame)
                                send_tf(tf_broadcaster, frame_id_l, child, pos_l, quat_l)
                        else:
                            # Too old: only diagnostics
                            diag["status"] = "no_person" if row["status"].startswith("SKIP:no_person") else "no_depth_image"
                            publish_diag(diag_pub, diag)
                    else:
                        # Never saw a valid frame yet: only diagnostics
                        diag["status"] = "no_person" if row["status"].startswith("SKIP:no_person") else "no_depth_image"
                        publish_diag(diag_pub, diag)

                rate.sleep()

        if not args.loop:
            break

if __name__ == "__main__":
    main()

# ---------------------- Usage examples ----------------------
# On a ROS machine (ROS1):
#   source /opt/ros/noetic/setup.bash
#   roscore
#
# Replay at 10 Hz, PoseStamped + diagnostics only:
#   python demo/replay_pose.py --log demo/output/color_0141/pose_stamped.log --hz 10
#
# Replay + RViz Marker + TF frame:
#   python demo/replay_pose.py --log demo/output/color_0141/pose_stamped.log --hz 10 \
#       --publish-marker --marker-topic /human_pose_marker --marker-scale 0.2 \
#       --publish-tf --tf-child-frame human_base
#
# Inspect: 
#   rostopic echo /human_pose -n 3
#   rostopic echo /human_pose_status -n 3
#   rostopic hz /human_pose
#
# In RViz:
#   - Fixed Frame: camera_color_optical_frame
#   - Add 'TF' display (to see the human_base frame)
#   - Add 'Marker' display on /human_pose_marker (arrow at the pose)
