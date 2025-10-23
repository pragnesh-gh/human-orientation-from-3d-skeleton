#!/usr/bin/env python3
import argparse, re, time, json
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

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

def commit_frame(best_row, lkv, pose_pub, diag_pub, timeout_ms):
    """
    Commit a single, best line for the current frame:
      - If row is OK with valid pos, publish fresh and update LKV.
      - Else, try to republish last-known if still fresh (stale).
      - Emit a diagnostics String JSON either way.
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
    args = ap.parse_args()

    rospy.init_node("pose_replayer", anonymous=True)
    pose_pub = rospy.Publisher(args.topic, PoseStamped, queue_size=10)
    diag_pub = rospy.Publisher(args.diag_topic, String, queue_size=10)
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
                    commit_frame(best_row, lkv, pose_pub, diag_pub, timeout)
                    best_row = None
                    rate.sleep()

                current_frame = row["frame_idx"]
                best_row = better(best_row, row)

            # EOF: commit the last buffered frame
            if not rospy.is_shutdown():
                commit_frame(best_row, lkv, pose_pub, diag_pub, timeout)
                rate.sleep()

        if not args.loop or rospy.is_shutdown():
            break

if __name__ == "__main__":
    main()
