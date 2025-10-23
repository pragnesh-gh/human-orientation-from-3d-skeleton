#!/usr/bin/env python3
import argparse, re, time, json
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

LINE_RE = re.compile(
    r'^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(\S+)\s*\|\s*([NA0-9.]+)\s*\|\s*([NA0-9.\-]+)\s*\|\s*([0-9eE.\-\s]+)\s*\|\s*([A-Z:]+)\s*\|\s*(\[.*?\]|NA)\s*\|\s*(\w+)\s*\|\s*(.*)\s*$'
)
# fields: frame_index | has_person | t_ms | frame_id | conf | yaw_deg | qw qx qy qz | status | pos_xyz_m | translation_source | note

def parse_line(line):
    if line.strip().startswith('#') or not line.strip():
        return None
    m = LINE_RE.match(line)
    if not m: return None
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
        pos_xyz = nums[:3] if len(nums)>=3 else None
    trans_src = m.group(10)
    note = m.group(11)
    return {
        "frame_idx": frame_idx, "has_person": has_person, "t_ms": t_ms, "frame_id": frame_id,
        "conf": conf, "yaw_deg": yaw_deg, "quat": qwqxqyqz, "status": status,
        "pos": pos_xyz, "translation_source": trans_src, "note": note
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to pose_stamped.log")
    ap.add_argument("--hz", type=float, default=10.0, help="publish rate")
    ap.add_argument("--topic", default="/human_pose", help="PoseStamped topic")
    ap.add_argument("--diag-topic", default="/human_pose_status", help="diagnostics topic (String JSON)")
    ap.add_argument("--stale-timeout-ms", type=int, default=1000)
    ap.add_argument("--loop", action="store_true", help="loop the log")
    args = ap.parse_args()

    rospy.init_node("pose_replayer", anonymous=True)
    pose_pub = rospy.Publisher(args.topic, PoseStamped, queue_size=10)
    diag_pub = rospy.Publisher(args.diag_topic, String, queue_size=10)
    rate = rospy.Rate(args.hz)

    # last-known valid
    lkv = {"pose": None, "t_wall_ms": None}
    timeout = args.stale_timeout_ms

    def publish_pose(frame_id, pos, quat_wxyz):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        # ROS uses x,y,z,w
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
        msg.pose.orientation.x = quat_wxyz[1]
        msg.pose.orientation.y = quat_wxyz[2]
        msg.pose.orientation.z = quat_wxyz[3]
        msg.pose.orientation.w = quat_wxyz[0]
        pose_pub.publish(msg)

    while not rospy.is_shutdown():
        with open(args.log, "r", encoding="utf-8") as f:
            for line in f:
                if rospy.is_shutdown(): break
                row = parse_line(line)
                if not row:
                    continue

                frame_id = row["frame_id"]
                conf = row["conf"] if row["conf"] is not None else 1.0
                status = row["status"]
                trans_src = row["translation_source"]
                note = row["note"]
                now_ms = int(time.time()*1000)

                diag = {
                    "status": None, "confidence": round(conf,3),
                    "translation_source": trans_src, "note": note,
                    "frame_id": frame_id, "age_ms": 0
                }

                if row["pos"] is not None:
                    # fresh valid frame
                    pos = tuple(row["pos"])
                    quat = tuple(row["quat"])
                    publish_pose(frame_id, pos, quat)
                    lkv["pose"] = (frame_id, pos, quat)
                    lkv["t_wall_ms"] = now_ms
                    diag["status"] = "ok"
                    diag["x"], diag["y"], diag["z"] = pos
                else:
                    # invalid: republish last-known if fresh enough
                    if lkv["pose"] is not None:
                        age = now_ms - (lkv["t_wall_ms"] or now_ms)
                        diag["age_ms"] = int(age)
                        if age <= timeout:
                            frame_id_l, pos_l, quat_l = lkv["pose"]
                            publish_pose(frame_id_l, pos_l, quat_l)
                            diag["status"] = "stale"
                        else:
                            diag["status"] = "no_person" if row["status"].startswith("SKIP:no_person") else "no_depth_image"
                    else:
                        diag["status"] = "no_person" if row["status"].startswith("SKIP:no_person") else "no_depth_image"

                diag_pub.publish(String(data=json.dumps(diag)))
                rate.sleep()
        if not args.loop:
            break

if __name__ == "__main__":
    main()
