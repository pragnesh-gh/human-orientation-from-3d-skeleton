#!/usr/bin/env python3
import os, argparse, math
from collections import deque  # not needed anymore, but harmless
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import glob, re

def sec_from_stamp(stamp):
    return stamp.secs + stamp.nsecs * 1e-9

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_color(path, cv_bgr):
    cv2.imwrite(path, cv_bgr)

def save_depth(path, cv_u16):
    if cv_u16.dtype != np.uint16:
        cv_u16 = cv_u16.astype(np.uint16)
    cv2.imwrite(path, cv_u16)

def nearest_match(src_stamp, queue, tol_sec):
    """queue is a list of (t, img). Remove and return the closest within tol."""
    if not queue:
        return None
    best_i, best_dt = -1, float('inf')
    for i, (t, _) in enumerate(queue):
        dt = abs(t - src_stamp)
        if dt < best_dt:
            best_dt = dt
            best_i = i
    if best_i >= 0 and best_dt <= tol_sec:
        return queue.pop(best_i)  # works for list
    return None

# --- helper to auto-continue indexing ---
def _max_index_in(dir_path, prefix):
    pat = os.path.join(dir_path, f"{prefix}_*.png")
    mx = -1
    for p in glob.glob(pat):
        m = re.search(rf"{prefix}_(\d+)\.png$", os.path.basename(p))
        if m:
            mx = max(mx, int(m.group(1)))
    return mx

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
    ap.add_argument("--start-index", type=int, default=0,
                   help="filename index offset; use -1 to auto-continue from existing files")
    args = ap.parse_args()

    color_dir = os.path.join(args.out, "color")
    depth_dir = os.path.join(args.out, "depth")
    ensure_dir(color_dir)
    ensure_dir(depth_dir)

    start_idx = args.start_index
    if start_idx < 0:
        # auto-continue from whichever side is higher (color or depth)
        mx_c = _max_index_in(color_dir, "color")
        mx_d = _max_index_in(depth_dir, "depth")
        start_idx = max(mx_c, mx_d) + 1
        print(f"[INFO] auto start-index = {start_idx}")

    bridge = CvBridge()
    # --- use lists (not deque) so we can pop by index
    color_q = []  # entries: (t, cv_bgr)
    depth_q = []  # entries: (t, cv_u16)

    pairs_saved = 0
    seen_pairs = 0
    last_pair_rel = None
    total_sec = None

    with rosbag.Bag(args.bag, "r") as bag:
        t0 = bag.get_start_time()
        t1 = bag.get_end_time()
        total_sec = t1 - t0
        t_start = t0 + args.start_secs
        t_end = math.inf if args.duration is None else (t_start + args.duration)
        print(f"[INFO] bag window: {args.start_secs:.3f}s → "
              f"{(min(t_end,t1)-t0):.3f}s of total {total_sec:.3f}s")

        topics = [args.color_topic, args.depth_topic]
        for topic, msg, stamp in bag.read_messages(topics=topics):
            t = stamp.secs + stamp.nsecs * 1e-9
            if t < t_start:
                continue
            if t > t_end:
                break

            if topic == args.color_topic:
                try:
                    cv_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except Exception:
                    continue
                match = nearest_match(t, depth_q, args.time_tol)
                if match is not None:
                    t_depth, cv_u16 = match
                    seen_pairs += 1
                    if (seen_pairs - 1) % args.stride == 0:
                        idx = start_idx + pairs_saved
                        save_color(os.path.join(color_dir, f"color_{idx:06d}.png"), cv_bgr)
                        save_depth(os.path.join(depth_dir,  f"depth_{idx:06d}.png"),  cv_u16)
                        pairs_saved += 1
                        if pairs_saved % 50 == 0:
                            print(f"[INFO] saved pairs: {pairs_saved}")
                        if pairs_saved >= args.max_pairs:
                            break
                    last_pair_rel = (t if topic == args.color_topic else t_depth) - t0
                else:
                    color_q.append((t, cv_bgr))
                    if len(color_q) > 200:
                        del color_q[0]

            elif topic == args.depth_topic:
                try:
                    cv_u16 = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                except Exception:
                    continue
                if cv_u16.dtype != np.uint16:
                    cv_u16 = cv_u16.astype(np.uint16)

                match = nearest_match(t, color_q, args.time_tol)
                if match is not None:
                    t_color, cv_bgr = match
                    seen_pairs += 1
                    if (seen_pairs - 1) % args.stride == 0:
                        idx = start_idx + pairs_saved
                        save_color(os.path.join(color_dir, f"color_{idx:06d}.png"), cv_bgr)
                        save_depth(os.path.join(depth_dir,  f"depth_{idx:06d}.png"),  cv_u16)
                        pairs_saved += 1
                        if pairs_saved % 50 == 0:
                            print(f"[INFO] saved pairs: {pairs_saved}")
                        if pairs_saved >= args.max_pairs:
                            break
                    last_pair_rel = (t_color if topic == args.color_topic else t) - t0
                else:
                    depth_q.append((t, cv_u16))
                    if len(depth_q) > 200:
                        del depth_q[0]

    if last_pair_rel is not None and total_sec is not None:
        print(f"[PROGRESS] last_pair_at = {last_pair_rel:.3f}s / {total_sec:.3f}s")
        print(f"[HINT] next run could use: --start-secs {last_pair_rel:.3f} "
              f"--start-index -1 (auto-continue)")
    print(f"[DONE] saved {pairs_saved} pairs to {args.out}/(color|depth)")

if __name__ == "__main__":
    main()
