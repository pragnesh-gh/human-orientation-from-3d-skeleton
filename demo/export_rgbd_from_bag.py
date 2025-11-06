#!/usr/bin/env python3
import os, argparse, math
from collections import deque
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

def sec_from_stamp(stamp):
    return stamp.secs + stamp.nsecs * 1e-9

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_color(path, cv_bgr):
    cv2.imwrite(path, cv_bgr)

def save_depth(path, cv_u16):
    # Ensure uint16, write as 16-bit PNG
    if cv_u16.dtype != np.uint16:
        cv_u16 = cv_u16.astype(np.uint16)
    cv2.imwrite(path, cv_u16)

def nearest_match(src_stamp, queue, tol_sec):
    """Find and remove from queue the element with stamp nearest to src_stamp within tol_sec."""
    if not queue:
        return None
    # Linear scan is fine with small buffers; keep buffers short.
    best_i, best_dt = -1, float('inf')
    for i, (t, _) in enumerate(queue):
        dt = abs(t - src_stamp)
        if dt < best_dt:
            best_dt = dt
            best_i = i
    if best_i >= 0 and best_dt <= tol_sec:
        return queue.pop(best_i)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="path to input rosbag")
    ap.add_argument("--out", required=True, help="output root (will create color/ and depth/)")
    ap.add_argument("--color-topic", default="/camera/color/image_raw")
    ap.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    ap.add_argument("--max-pairs", type=int, default=500, help="stop after this many pairs")
    ap.add_argument("--start-secs", type=float, default=0.0, help="offset from bag start (seconds)")
    ap.add_argument("--duration", type=float, default=None, help="limit time window length (seconds)")
    ap.add_argument("--time-tol", type=float, default=0.03, help="pairing tolerance (seconds)")
    ap.add_argument("--stride", type=int, default=1, help="save every Nth pair (subsample)")
    args = ap.parse_args()

    color_dir = os.path.join(args.out, "color")
    depth_dir = os.path.join(args.out, "depth")
    ensure_dir(color_dir)
    ensure_dir(depth_dir)

    bridge = CvBridge()
    color_q = deque()  # entries: (t, cv_bgr)
    depth_q = deque()  # entries: (t, cv_u16)

    pairs_saved = 0
    seen_pairs = 0

    with rosbag.Bag(args.bag, "r") as bag:
        t0 = bag.get_start_time()
        t_start = t0 + args.start_secs
        t_end = math.inf if args.duration is None else (t_start + args.duration)

        # Restrict to the two topics for speed
        topics = [args.color_topic, args.depth_topic]
        for topic, msg, stamp in bag.read_messages(topics=topics):
            t = sec_from_stamp(stamp)
            if t < t_start:
                continue
            if t > t_end:
                break

            if topic == args.color_topic:
                # Convert to BGR8
                try:
                    cv_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except Exception as e:
                    # Skip on conversion errors
                    continue
                # Try to match with pending depth
                match = nearest_match(t, depth_q, args.time_tol)
                if match is not None:
                    t_depth, cv_u16 = match
                    seen_pairs += 1
                    if (seen_pairs - 1) % args.stride == 0:
                        idx = pairs_saved
                        color_path = os.path.join(color_dir, f"color_{idx:06d}.png")
                        depth_path = os.path.join(depth_dir, f"depth_{idx:06d}.png")
                        save_color(color_path, cv_bgr)
                        save_depth(depth_path, cv_u16)
                        pairs_saved += 1
                        if pairs_saved % 50 == 0:
                            print(f"[INFO] saved pairs: {pairs_saved}")
                        if pairs_saved >= args.max_pairs:
                            break
                else:
                    # Buffer color until a matching depth arrives
                    color_q.append((t, cv_bgr))
                    # keep buffer short
                    while len(color_q) > 200:
                        color_q.popleft()

            elif topic == args.depth_topic:
                # Convert to raw uint16 depth
                try:
                    cv_u16 = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                except Exception as e:
                    continue
                if cv_u16.dtype != np.uint16:
                    # Try forcing to uint16 if it's mono16 as int16 etc.
                    cv_u16 = cv_u16.astype(np.uint16)

                match = nearest_match(t, color_q, args.time_tol)
                if match is not None:
                    t_color, cv_bgr = match
                    seen_pairs += 1
                    if (seen_pairs - 1) % args.stride == 0:
                        idx = pairs_saved
                        color_path = os.path.join(color_dir, f"color_{idx:06d}.png")
                        depth_path = os.path.join(depth_dir, f"depth_{idx:06d}.png")
                        save_color(color_path, cv_bgr)
                        save_depth(depth_path, cv_u16)
                        pairs_saved += 1
                        if pairs_saved % 50 == 0:
                            print(f"[INFO] saved pairs: {pairs_saved}")
                        if pairs_saved >= args.max_pairs:
                            break
                else:
                    depth_q.append((t, cv_u16))
                    while len(depth_q) > 200:
                        depth_q.popleft()

    print(f"[DONE] saved {pairs_saved} pairs to {args.out}/(color|depth)")

if __name__ == "__main__":
    main()
