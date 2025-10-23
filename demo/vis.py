import argparse
import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from lib.preprocess import h36m_coco_format, revise_kpts
# from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

POSE_DEBUG = os.getenv("POSE_DEBUG", "0") != "0"


try:
    from common.orientation import compute_torso_frame, smooth_quat
except ModuleNotFoundError:
    # As a fallback, also try the current working directory (in case you run from repo root)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    from common.orientation import compute_torso_frame, smooth_quat
import json, csv
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
import shutil
from pathlib import Path
from contextlib import contextmanager
import math
import re
import time

def _dbg(msg, **kwargs):
    print(f"[VIS.DBG] {msg}")
    if kwargs:
        for k, v in kwargs.items():
            print(f"  - {k}: {v}")

@contextmanager
def _strip_argv():
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        yield
    finally:
        sys.argv = saved
# --- PoseStamped logging helpers (minimal & off-by-default) ---
def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _open_pose_log(log_path: str):
    """
    Create/overwrite the PoseStamped-style log file and write a short schema header.
    Human-readable: one line per frame.
    """
    _ensure_dir(log_path)
    need_header = (not os.path.exists(log_path)) or (os.path.getsize(log_path) == 0)
    f = open(log_path, "a", encoding="utf-8")  # append
    if need_header:
        f.write("# PoseStamped log (one line per frame)\n")
        f.write("# fields: frame_index | has_person | t_ms | frame_id | "
                "conf | yaw_deg | qw qx qy qz | status | pos_xyz_m | translation_source | note\n")
    return f

def _fmt(x):
    return "NA" if x is None else (f"{float(x):.6f}" if isinstance(x, (int, float)) else str(x))

def _yaw_deg_from_forward(fwd):
    """
    Optical camera frame: x=right, y=down, z=forward.
    yaw = atan2(fx, fz) in degrees (rotation about 'up', ignoring roll/pitch).
    """
    fx, fy, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])
    return math.degrees(math.atan2(fx, fz + 1e-8))

def _log_pose_stamped(
    fh,
    *,
    frame_i: int,
    has_person: bool,
    t_ms: float,
    frame_id: str,
    conf=None,
    yaw_deg=None,
    quat_wxyz=None,          # [w,x,y,z] or None
    status: str = "OK",
    pos_xyz_m=None,          # [x,y,z] in meters or None
    translation_source: str = "none",
    note: str = ""
):
    if quat_wxyz is None:
        quat_wxyz = [0.0, 0.0, 0.0, 0.0]
    line = (
        f"{frame_i:06d} | {int(has_person)} | {t_ms:.2f} | {frame_id} | "
        f"{_fmt(conf)} | {_fmt(yaw_deg)} | "
        f"{_fmt(quat_wxyz[0])} {_fmt(quat_wxyz[1])} {_fmt(quat_wxyz[2])} {_fmt(quat_wxyz[3])} | "
        f"{status} | {_fmt(pos_xyz_m)} | {translation_source} | {note}\n"
    )
    fh.write(line)
# ==== Depth / Intrinsics helpers =================================================
def _read_depth_png(path):
    """Read aligned 16-bit depth PNG. Returns np.ndarray or None."""
    if not path:
        return None
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # Expect 16UC1; if something else sneaks in, still return it for debugging
    return img

def _median_depth_at(depth_u16, u_px, v_px, ks=7):
    """Median depth (in raw units) in a ks×ks window around (u,v). Ignores zeros."""
    if depth_u16 is None:
        return None
    h, w = depth_u16.shape[:2]
    u = int(np.clip(round(u_px), 0, w - 1))
    v = int(np.clip(round(v_px), 0, h - 1))
    r = ks // 2
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch = depth_u16[v0:v1, u0:u1].astype(np.int64)
    if patch.size == 0:
        return None
    nz = patch[patch > 0]
    if nz.size == 0:
        return None
    return int(np.median(nz))

def _backproject_uvz_to_xyz(u, v, z_m, fx, fy, cx, cy):
    """
    Pinhole back-projection (camera optical frame: +x right, +y down, +z forward).
    Inputs: pixel (u,v), depth in meters, intrinsics.
    """
    x_m = (u - cx) / fx * z_m
    y_m = (v - cy) / fy * z_m
    return x_m, y_m, z_m


def _hrnet_safe_image_path(image_path: str, tmp_dir: str) -> str:
    """
    Return a literal file path safe for HRNet:
    - Copies the image to tmp_dir/single.png if the basename ends with digits (…_0661.png),
      to prevent 'sequence' heuristics.
    - Normalizes to forward slashes to avoid Windows FFmpeg hiccups.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    p = image_path
    basename = os.path.basename(p)
    # looks like ..._1234.png or ...0001.png etc.
    if re.search(r'\d+\.png$', basename, flags=re.IGNORECASE):
        tmp_path = os.path.join(tmp_dir, "single.png")
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        cv2.imwrite(tmp_path, img)
        return tmp_path.replace("\\", "/")
    return p.replace("\\", "/")
# ========== LKV + ROS helpers (safe on Windows) ==========
_LKV = {
    "pose": None,        # dict: {"x":..,"y":..,"z":..,"qx":..,"qy":..,"qz":..,"qw":..}
    "t_ms": None         # last fresh timestamp (monotonic ms or your pos_ms)
}

def _now_ms():
    return int(time.time() * 1000)

# Lazy ROS init + publishers (only if --ros-publish and ROS is installed)
_ros_ready = False
_ros_pose_pub = None
_ros_diag_pub = None

def _try_init_ros(args):
    global _ros_ready, _ros_pose_pub, _ros_diag_pub
    if _ros_ready:
        return
    if not getattr(args, "ros_publish", False):
        return
    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
        if not rospy.core.is_initialized():
            rospy.init_node('poseformer_publisher', anonymous=True, disable_signals=True)
        _ros_pose_pub = rospy.Publisher(args.ros_topic, PoseStamped, queue_size=10)
        _ros_diag_pub = rospy.Publisher(args.ros_diag_topic, String, queue_size=10)
        _ros_ready = True
        print(f"[ROS] Publishers ready: {args.ros_topic}, {args.ros_diag_topic}")
    except Exception as e:
        print(f"[ROS] Not available (won't publish). Reason: {e}")
        _ros_ready = False

def _publish_ros_pose(frame_id, pos, quat, stamp_now=True):
    if not _ros_ready or pos is None or quat is None:
        return
    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = rospy.Time.now() if stamp_now else rospy.Time()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
        # ROS uses x,y,z,w order
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat[1], quat[2], quat[3], quat[0]
        _ros_pose_pub.publish(msg)
    except Exception as e:
        print(f"[ROS] Pose publish error: {e}")

def _publish_ros_diag(diag_dict):
    if not _ros_ready:
        return
    try:
        from std_msgs.msg import String
        _ros_diag_pub.publish(String(data=json.dumps(diag_dict)))
    except Exception as e:
        print(f"[ROS] Diag publish error: {e}")

def _emit_diag_console(diag_dict):
    # Compact one-line console emitter
    try:
        print(f"[DIAG] {json.dumps(diag_dict, separators=(',', ':'))}")
    except Exception:
        pass





sys.path.append(os.getcwd())
from common.model_poseformer import PoseTransformerV2 as Model
from common.camera import *
from common.camera import normalize_screen_coordinates  # for 2D norm (w,h) → [-1,1]
from common.generators import UnchunkedGenerator       # to handle padding/windowing

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec



plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# suppress warnings while we ALSO properly close figures everywhere
matplotlib.rcParams['figure.max_open_warning'] = 0

def show2Dpose(kps, img):
    # Expect a single frame worth of 2D keypoints (17 joints, x/y)
    if not (isinstance(kps, np.ndarray) and kps.shape == (17, 2)):
        raise ValueError(f"show2Dpose expected kps shape (17,2), got {getattr(kps, 'shape', None)}")

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, (a, b) in enumerate(connections):
        p0 = np.asarray(kps[a], dtype=np.int32).ravel()
        p1 = np.asarray(kps[b], dtype=np.int32).ravel()
        start = (int(p0[0]), int(p0[1]))
        end = (int(p1[0]), int(p1[1]))
        cv2.line(img, start, end, lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, start, radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, end, radius=3, color=(0, 255, 0), thickness=-1)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


# def get_pose2D(source_path, output_dir):
#     # cap = cv2.VideoCapture(video_path)
#     # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
#     print('\nGenerating 2D pose...')
#
#
#     #importing here because its classing with the arg parse with image.
#     saved_argv = sys.argv
#     try:
#         sys.argv = [saved_argv[0]]  # strip all external CLI flags
#         from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
#     finally:
#         sys.argv = saved_argv
#
#     keypoints, scores = hrnet_pose(source_path, det_dim=416, num_peroson=1, gen_output=True)
#     keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
#     re_kpts = revise_kpts(keypoints, scores, valid_frames)
#     print('Generating 2D pose successful!')
#
#     output_dir += 'input_2D/'
#     os.makedirs(output_dir, exist_ok=True)
#
#     output_npz = output_dir + 'keypoints.npz'
#     np.savez_compressed(output_npz, reconstruction=keypoints)

def img2video(video_path, output_dir):
    _dbg("Entering img2video", video_path=video_path, output_dir=output_dir)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25  # fallback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    if not names:
        _dbg("img2video: no pose frames found", dir=output_dir + 'pose/')
        return

    img0 = cv2.imread(names[0])
    size = (img0.shape[1], img0.shape[0])

    source_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{source_name}.mp4")
    vw = cv2.VideoWriter(out_path, fourcc, fps, size)

    for name in names:
        frame = cv2.imread(name)
        vw.write(frame)

    vw.release()
    cap.release()
    _dbg("img2video: wrote", path=out_path, fps=fps, nframes=len(names))




def get_pose2D(source_path, output_dir):
    _dbg("Entering get_pose2D", source_path=source_path, output_dir=output_dir)

    print('\nGenerating 2D pose...')

    # --- Shield HRNet import from our CLI flags ---
    with _strip_argv():
        _dbg("Importing hrnet gen_kpts with stripped argv", argv=sys.argv)
        from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
    _dbg("Imported hrnet gen_kpts", func_module=hrnet_pose.__module__, func_name=hrnet_pose.__name__)

    # Small helper to write a fallback npz + log when no person / HRNet fails
    def _write_fallback_npz_and_log():
        # Determine T (frames)
        T = 1
        # Try to detect if it's a video by opening with cv2 and reading frame count
        try:
            cap = cv2.VideoCapture(source_path)
            if cap is not None and cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frames and frames > 0:
                    T = frames
            if cap is not None:
                cap.release()
        except Exception:
            pass

        # Build zeros reconstruction: (1, T, 17, 2) so downstream code is happy
        reconstruction = np.zeros((1, T, 17, 2), dtype=np.float32)
        valid_mask = np.zeros((T,), dtype=np.uint8)

        out2d_dir = os.path.join(output_dir, 'input_2D')
        os.makedirs(out2d_dir, exist_ok=True)
        output_npz = os.path.join(out2d_dir, 'keypoints.npz')
        np.savez_compressed(output_npz, reconstruction=reconstruction, valid_mask=valid_mask)

        # Write a simple log so you know this source was skipped for pose detection
        skip_log = os.path.join(output_dir, 'skip.log')
        with open(skip_log, 'a', encoding='utf-8') as f:
            f.write(f"[SKIP] No person detected or HRNet failed for: {source_path}\n")
            f.write(f"       Wrote fallback npz to: {output_npz}\n")
        _dbg("Wrote fallback npz + skip log", frames=T, npz=output_npz, log=skip_log)

    # --- Shield HRNet call from our CLI flags as well ---
    try:
        with _strip_argv():
            tmp_hrnet_dir = os.path.join(output_dir, "input_2D", "_hrnet_tmp")
            hrnet_src = (
                _hrnet_safe_image_path(source_path, tmp_hrnet_dir)
                if is_image
                else source_path.replace("\\", "/")
            )
            _dbg("Calling hrnet_pose(...) with stripped argv", argv=sys.argv,
                 det_dim=416, num_peroson=1, gen_output=True)
            keypoints, scores = hrnet_pose(hrnet_src, det_dim=416, num_peroson=1, gen_output=True)
    except Exception as e:
        # Typical when no people are present: ValueError from internal transpose
        _dbg("HRNet call failed with Exception; writing fallback npz and continuing", err=str(e))
        _write_fallback_npz_and_log()
        print('Generating 2D pose (fallback, no person) complete!')
        return

    _dbg("HRNet returned", kpts_shape=np.array(keypoints).shape, scores_shape=np.array(scores).shape)

    try:
        keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        _dbg("After h36m_coco_format", kpts_shape=np.array(keypoints).shape,
             scores_shape=np.array(scores).shape, valid_frames_len=len(valid_frames))

        re_kpts = revise_kpts(keypoints, scores, valid_frames)
        _dbg("After revise_kpts", re_kpts_shape=np.array(re_kpts).shape)

        # --- If HRNet produced no usable frames/person, write fallback and stop 2D ---
        if (not isinstance(re_kpts, np.ndarray)) or re_kpts.size == 0 or len(valid_frames) == 0:
            _dbg("Empty 2D detections; writing fallback npz",
                 re_kpts_shape=str(getattr(re_kpts, "shape", None)), valid_frames_len=len(valid_frames))
            _write_fallback_npz_and_log()
            print('Generating 2D pose (no person) complete!')
            return

        print('Generating 2D pose successful!')

        out2d_dir = os.path.join(output_dir, 'input_2D')
        os.makedirs(out2d_dir, exist_ok=True)
        output_npz = os.path.join(out2d_dir, 'keypoints.npz')

        # Build a frame-aligned valid mask from valid_frames (no dropped frames)
        T = int(re_kpts.shape[1]) if re_kpts.ndim >= 2 else len(valid_frames)
        valid_mask = np.zeros((T,), dtype=np.uint8)
        if len(valid_frames) > 0:
            vf = np.array(valid_frames, dtype=int)
            vf = vf[(vf >= 0) & (vf < T)]
            valid_mask[vf] = 1

        np.savez_compressed(
            output_npz,
            reconstruction=re_kpts,       # aligned to full video length
            valid_mask=valid_mask         # 1 = person present on that frame
        )
        if POSE_DEBUG:
            print(f"[2D.DBG] Saved {output_npz}: T={re_kpts.shape[1]}, valid_sum={int(valid_mask.sum())}")
        _dbg("Saved 2D keypoints", output_npz=output_npz)
    except Exception as e:
        # If post-HRNet formatting fails for an empty/odd case, fall back too
        _dbg("Post-processing failed; writing fallback npz and continuing", err=str(e))
        _write_fallback_npz_and_log()
        print('Generating 2D pose (fallback, post-process) complete!')
        return



def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(video_path, output_dir, is_image=False, args=None):
    """
    Run PoseFormerV2 3D reconstruction (and optional torso orientation) on a video or single image.

    Outputs (under <output_dir>/):
      - pose2D/  : per-frame 2D overlays (PNG)
      - pose3D/  : per-frame 3D plots (PNG)
      - pose/    : per-frame side-by-side composites (PNG)
      - pose_stamped.log (optional): PoseStamped-style text log
      - orientation {csv|json} (optional): per-frame torso orientation table

    Parameters
    ----------
    video_path : str
        Path to input video or image.
    output_dir : str
        Directory to write outputs into.
    is_image : bool
        True if `video_path` is a single still image.
    args : argparse.Namespace or simple object
        Must expose any CLI toggles used in this function (see inline getattr(...) calls).
        If None, a minimal dummy is created and defaults are applied.
    """

    # args, _ = argparse.ArgumentParser().parse_known_args()
    _dbg("Entering get_pose3D", video_path=video_path, output_dir=output_dir, is_image=is_image)

    # ----------------------------
    # 0) Args & constants
    # ----------------------------
    if args is None:
        class Dummy: pass
        args = Dummy()

    # Core PoseFormerV2 model params (kept as your working defaults)
    args.embed_dim_ratio, args.depth, args.frames = 32, 4, 243
    args.number_of_kept_frames, args.number_of_kept_coeffs = 27, 27
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/'
    args.n_joints, args.out_joints = 17, 17

    # Output folders
    output_dir_2D = os.path.join(output_dir, "pose2D/")
    output_dir_3D = os.path.join(output_dir, "pose3D/")
    os.makedirs(output_dir_2D, exist_ok=True)
    os.makedirs(output_dir_3D, exist_ok=True)

    # ----------------------------
    # 1) Load model (strict load)
    # ---------------------------
    ## Reload
    model = nn.DataParallel(Model(args=args)).cuda()
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '27_243_45.2.bin')))[0]
    _dbg("Model & checkpoint", model_path=model_path)
    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict['model_pos'], strict=True)
    model.eval()


    # ----------------------------
    # 2) Load 2D inputs (+ valid_mask if present)
    # ----------------------------
    _npz = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)
    keypoints = _npz['reconstruction']                                                  # shape ~ (M=1, T, 17, 2) in pixels
    valid_mask = _npz['valid_mask'] if 'valid_mask' in _npz.files else None             # (T,) 1=person present


    # --- Logging priority helper: choose one line per frame ---
      # Higher number = higher priority when choosing the single line to emit
    _STATUS_RANK = {
            "OK": 5,
            "SKIP:low_confidence(thresh=": 3,  # prefix match handled below
            "SKIP:orientation_disabled": 2,
            "SKIP:no_depth": 1,
            "SKIP:no_person": 0,
    }

    def _status_rank(s: str) -> int:
        if s in _STATUS_RANK:
            return _STATUS_RANK[s]
        # Handle dynamic notes like SKIP:low_confidence(thresh=0.50)
        if s.startswith("SKIP:low_confidence("):
            return _STATUS_RANK["SKIP:low_confidence(thresh="]
        return -1

    def _stage_best(best, candidate):
        """
            best: dict or None
            candidate: dict with keys:
              frame_i, has_person, t_ms, frame_id, conf, yaw_deg, quat_wxyz,
              status, pos_xyz_m, translation_source, note
            Returns the better (higher priority) dict.
        """

        if candidate is None:
            return best
        if best is None:
            return candidate
        br = _status_rank(best.get("status", ""))
        cr = _status_rank(candidate.get("status", ""))
        return candidate if cr > br else best
    # ----------------------------
    # 3) Open source (video or image)
    # ----------------------------

    cap = None
    if is_image:
        img0 = cv2.imread(video_path)
        if img0 is None:
            raise FileNotFoundError(f"Cannot read image: {video_path}")
        video_length = 1
        base_h, base_w = img0.shape[:2]
        fps, pos_ms = 0.0, 0.0
    else:
        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        base_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        base_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        if base_w <= 0 or base_h <= 0:
            base_w, base_h = 640, 480  # safe fallback


    # ----------------------------
    # 4) PoseStamped logging (only if enabled)
    # ----------------------------
    pose_log_fp = None
    pose_log_path = None
    enable_pose_log = bool(getattr(args, "log_pose_stamped", False) or getattr(args, "ros_log_file", None))

    if (not isinstance(keypoints, np.ndarray)) or keypoints.size == 0:
        # Write a single SKIP line and exit gracefully
        if pose_log_fp is not None:
            _log_pose_stamped(
                pose_log_fp,
                frame_i=0, has_person=False, t_ms=0.0,
                frame_id=getattr(args, "ros_frame_id", "camera_color_optical_frame"),
                conf=None, yaw_deg=None, quat_wxyz=None,
                status="SKIP:no_2d_keypoints",
                pos_xyz_m=None, translation_source="none",
                note="2D stage returned empty"
            )
        return

    if enable_pose_log:
        # Path priority: --ros-log-file > --pose-log-file > <output_dir>/pose_stamped.log
        pose_log_path = (getattr(args, "ros_log_file", None)
                         or getattr(args, "pose_log_file", None)
                         or os.path.join(output_dir, "pose_stamped.log"))

        pose_log_fp = _open_pose_log(pose_log_path)  # append-only now
        _dbg("PoseStamped logging enabled", path=pose_log_path)




    # ----------------------------
    # 5) Runtime state
    # ----------------------------
    last_quat_for_log = None  # updated whenever we compute orientation (if enabled)

    # --- Orientation buffers (only used if enabled) ---
    orientation_rows = []
    prev_quat = None
    prev_post_out = None # remember previous 3D pose to hold-through gaps

    if POSE_DEBUG:
        kp_T = keypoints.shape[1]
        print(f"[3D.DBG] video_len={video_length}, keypoints_T={kp_T}, "
              f"valid_mask_len={(len(valid_mask) if valid_mask is not None else 'None')}")

    # ----------------------------
    # 6) Frame loop (3D inference)
    # ----------------------------

    # --- rolling state used by diagnostics/publish (must exist before the loop) ---
    pos_xyz_m = None  # last known translation (meters)
    prev_quat = None  # last known orientation quaternion (w, x, y, z)
    has_person = False  # last detection flag
    yaw_deg = None
    conf = 1.0
    trans_src = "none"
    trans_note = ""

    try:
        ## 3D
        print('\nGenerating 3D pose...')
        for i in tqdm(range(video_length)):
            best_line = None  # holds the single best PoseStamped line to emit for this frame
            skip_rest_of_frame = False
            # -- Read frame (or reuse single image)
            if is_image:
                img = img0
                fps = 0.0
                pos_ms = 0.0
            else:
                ok, img = cap.read()
                if not ok or img is None:
                    if POSE_DEBUG:
                        print(f"[3D.DBG] Frame {i}: cap.read() failed")
                    # still produce empty outputs to keep timeline
                    img = np.zeros((base_h, base_w, 3), dtype=np.uint8)
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
                pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC)) if fps > 0 else 0.0

            # -- Person presence (prevents “speed-up” across gaps)
            has_person = True
            if valid_mask is not None and i < len(valid_mask):
                has_person = bool(valid_mask[i]) if i < len(valid_mask) else False # if keypoints shorter than video (should not happen after patch), treat as no person

            # -- Build temporal window [i-pad, i+pad] with edge padding to args.frames
            # T_all = keypoints.shape[1]
            T_all = 1 if is_image else keypoints.shape[1]
            start = max(0, i - args.pad)
            end = min(i + args.pad, T_all - 1)
            input_2D_no = keypoints[0][start:end + 1] # (win, 17, 2)

            left_pad = max(0, args.pad - i)
            right_pad = max(0, (i + args.pad) - (T_all - 1))
            if input_2D_no.shape[0] != args.frames:
                input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), mode="edge")

            if POSE_DEBUG:
                print(f"[3D.DBG] Frame {i}: has_person={has_person} win=[{start},{end}] padL={left_pad} padR={right_pad}")

            # -- GAP: if no person, write plain 2D frame, hold last 3D pose (or zeros), log, and continue
            if not has_person:
                # Save plain 2D frame (no overlay)
                cv2.imwrite(output_dir_2D + f"{i:04d}_2D.png", img)

                # Hold last 3D pose or zeros
                post_out = prev_post_out if prev_post_out is not None else np.zeros((17, 3), dtype=np.float32)
                fig = plt.figure(figsize=(9.6, 5.4))
                gs = gridspec.GridSpec(1, 1)
                gs.update(wspace=-0.00, hspace=0.05)
                ax = plt.subplot(gs[0], projection='3d')
                show3Dpose(post_out, ax)
                output_dir_3D = output_dir + 'pose3D/'
                os.makedirs(output_dir_3D, exist_ok=True)
                plt.savefig(output_dir_3D + f"{i:04d}_3D.png", dpi=200, format='png', bbox_inches='tight')
                plt.clf()
                plt.close(fig)

                # --- PoseStamped log for gap frame ---
                # if pose_log_fp is not None:
                    # Use pos_ms if available; otherwise 0
                t_ms = pos_ms if (not is_image and fps > 0) else 0.0
                    # Translation: we hold previous 3D if available; else zeros (already in post_out)
                    # tx_ty_tz = prev_post_out[0]
                    # Quaternion: keep last known if any, else zeros
                    # quat = last_quat_for_log if last_quat_for_log is not None else np.zeros(4, dtype=np.float32)
                    # _log_pose_stamped(
                    #     pose_log_fp,
                    #     frame_i=i, has_person=False, t_ms=t_ms,
                    #     frame_id=getattr(args, "ros_frame_id", "camera_color_optical_frame"),
                    #     conf=None, yaw_deg=None, quat_wxyz=None,
                    #     status="SKIP:no_person",
                    #     pos_xyz_m=None, translation_source="none",
                    #     note="gap/no_person"
                    # )
                # Stage a GAP line and mark this frame to skip the rest of processing,
                # but DO NOT 'continue' — we still want to reach the per-frame COMMIT.
                t_ms = pos_ms if (not is_image and fps > 0) else 0.0
                best_line = _stage_best(best_line, dict(frame_i = i, has_person = False, t_ms = t_ms,
                    frame_id = getattr(args, "ros_frame_id", "camera_color_optical_frame"),
                    conf = None, yaw_deg = None, quat_wxyz = None,
                    status = "SKIP:no_person",
                    pos_xyz_m = None, translation_source = "none",
                    note = "gap/no_person"))
                # skip_rest_of_frame = True

                # Commit exactly one line for this frame, then skip heavy work
                if pose_log_fp is not None and best_line is not None:
                    _log_pose_stamped(
                            pose_log_fp,
                               frame_i = best_line["frame_i"],
                           has_person = best_line["has_person"],
                           t_ms = best_line["t_ms"],
                           frame_id = best_line["frame_id"],
                           conf = best_line["conf"],
                           yaw_deg = best_line["yaw_deg"],
                           quat_wxyz = best_line["quat_wxyz"],
                           status = best_line["status"],
                           pos_xyz_m = best_line["pos_xyz_m"],
                           translation_source = best_line["translation_source"],
                           note = best_line["note"],
                       )
                continue

            # -- Normalize to [-1,1] screen coords
            input_2D = normalize_screen_coordinates(input_2D_no, w=img.shape[1], h=img.shape[0])

            # -- Horizontal flip augmentation (exactly once) + joint swap
            joints_left = [4, 5, 6, 11, 12, 13]
            joints_right = [1, 2, 3, 14, 15, 16]
            input_2D_aug = input_2D.copy()
            input_2D_aug[:, :, 0] *= -1
            input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]

            # Shape to (1, 2, 243, 17, 2) for model
            input_2D_stack = np.stack([input_2D, input_2D_aug], axis=0)[np.newaxis, ...]
            input_2D_tensor = torch.from_numpy(input_2D_stack.astype("float32")).cuda()

            # -- Model forward (non-flip + flip, then unflip & average)
            output_3D_non_flip = model(input_2D_tensor[:, 0])  # [1,1,17,3]
            output_3D_flip = model(input_2D_tensor[:, 1])  # [1,1,17,3]
            output_3D_flip[:, :, :, 0] *= -1
            output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
            output_3D = (output_3D_non_flip + output_3D_flip) / 2
            output_3D[:, :, 0, :] = 0  # root at origin
            post_out = output_3D[0, 0].detach().cpu().numpy()  # (17,3)

            # -- Camera->world & depth lift (keep your static R)
            rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            prev_post_out = post_out  # remember for gap frames

            # -- Choose overlay source index to avoid early/late drift
            # overlay_src_idx = min(i, keypoints.shape[1] - 1)
            # overlay_2D_px = keypoints[0, overlay_src_idx]  # shape (17, 2)
            # Use the center-of-window for video, index 0 for single image
            overlay_2D_px = input_2D_no[0 if is_image else args.pad]

            if POSE_DEBUG:
                est_ms = (i / fps * 1000.0) if fps > 0 else 0.0
                print("[SYNC.DBG] "
                      f"i={i} start={start} end={end} padL={left_pad} padR={right_pad} "
                      f"overlay_src_idx=NOT_USED pos_ms={pos_ms:.2f} est_ms={est_ms:.2f} fps={fps:.2f}")

            # ---- Step 2: last-known + diagnostics + optional ROS publish ----
            _try_init_ros(args)

            conf_val = float(conf) if ('conf' in locals()) else 1.0
            frame_id = getattr(args, "ros_frame_id", "camera_color_optical_frame")
            now_ms = _now_ms()
            timeout_ms = int(getattr(args, "stale_timeout_ms", 1000))

            # Decide status + which pose to publish (fresh or last-known)
            status = None
            pose_to_pub = None  # (pos, quat) to publish on /human_pose
            age_ms = 0

            # Fresh only if we have a translation AND a quaternion
            if has_person and (pos_xyz_m is not None) and (prev_quat is not None):
                # Fresh, valid measurement this frame
                status = "ok"
                _LKV["pose"] = {
                    "x": float(pos_xyz_m[0]), "y": float(pos_xyz_m[1]), "z": float(pos_xyz_m[2]),
                    "qx": float(prev_quat[1]), "qy": float(prev_quat[2]), "qz": float(prev_quat[3]),
                    "qw": float(prev_quat[0])
                }
                _LKV["t_ms"] = now_ms
                pose_to_pub = (
                    (_LKV["pose"]["x"], _LKV["pose"]["y"], _LKV["pose"]["z"]),
                    (float(prev_quat[0]), float(prev_quat[1]), float(prev_quat[2]), float(prev_quat[3]))  # w,x,y,z
                )
            else:
                # Invalid this frame: no person or no metric depth
                status = "stale" if _LKV["pose"] is not None else ("no_person" if not has_person else ("no_orientation" if (pos_xyz_m is not None) else "no_depth_image"))
                if _LKV["pose"] is not None:
                    age_ms = now_ms - int(_LKV["t_ms"] or now_ms)
                    if age_ms <= timeout_ms:
                        # Republish last-known pose (last publish + diagnostics)
                        p = _LKV["pose"]
                        pose_to_pub = (
                            (p["x"], p["y"], p["z"]),
                            (p["qw"], p["qx"], p["qy"], p["qz"])  # w,x,y,z
                        )
                    else:
                        # Too old: stop publishing on core topic; still emit diagnostics
                        pass

            # Build diagnostics (console + optional ROS diag topic)
            diag = {
                "status": status,
                "confidence": round(conf_val, 3),
                "translation_source": trans_src,
                "note": trans_note,
                "age_ms": age_ms,
                "frame_id": frame_id
            }
            if status == "ok" and (pos_xyz_m is not None):
                diag.update({
                    "x": float(pos_xyz_m[0]), "y": float(pos_xyz_m[1]), "z": float(pos_xyz_m[2])
                })

            _emit_diag_console(diag)
            _publish_ros_diag(diag)

            # Publish PoseStamped on /human_pose only if we have something to publish
            if pose_to_pub is not None:
                pos_pub, quat_pub = pose_to_pub
                _publish_ros_pose(frame_id, pos_pub, quat_pub, stamp_now=True)
            #############################################################################################

            # -- Write 2D overlay
            image = show2Dpose(overlay_2D_px, copy.deepcopy(img))
            cv2.putText(image, f"frame={i}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.imwrite(output_dir_2D + f"{i:04d}_2D.png", image)

            # -- Prepare 3D figure (and optional orientation triad)
            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)


            # -- Optional: torso orientation (estimate + smooth + overlay + remember for logging)
            if getattr(args, "estimate_orientation", False):
                layout = getattr(args, "orientation_layout","h36m")  # Default H36M; for COCO-17 pass --orientation-layout coco17
                alpha = float(getattr(args, "orientation_alpha", 0.6)) # slerp-EMA
                conf_min = float(getattr(args, "orientation_conf_min", 0.5))

                #CAMERA-FRAME joints for orientation (read model output BEFORE world transform)
                cam_post = output_3D[0, 0].cpu().detach().numpy()  # x:right, y:down, z:forward (optical frame)

                # Compute torso frame (camera/world consistent with post_out)
                quat_cam_wxyz, fwd_cam, right_cam, up_cam, conf = compute_torso_frame(cam_post, layout=layout)
                conf = 0.0 if conf is None else float(conf)

                # Confidence-gated SLERP-EMA
                cur_quat = quat_cam_wxyz
                if conf < conf_min and prev_quat is not None:
                    smoothed = prev_quat
                else:
                    smoothed = smooth_quat(prev_quat, cur_quat, alpha) if prev_quat is not None else cur_quat
                prev_quat = smoothed

                # Update cached quaternion for future logs
                last_quat_for_log = prev_quat

                # (Optional) yaw for logging if your logger supports it:
                yaw_deg = _yaw_deg_from_forward(fwd_cam)  # requires helper; safe to keep unused if not logging yaw

                #WORLD transform ONLY for visualization/rendering
                post_out = cam_post.copy()  # <- we branch here: cam_post stays camera-frame; post_out becomes world
                rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
                               dtype='float32')
                post_out = camera_to_world(post_out, R=rot, t=0)
                post_out[:, 2] -= np.min(post_out[:, 2])
                prev_post_out = post_out


                # Optional overlay on the same 3D axes
                if getattr(args, "orientation_overlay", False):
                    # Recompute orientation on the world-coordinates skeleton for a correct triad in that space
                    quat_wxyz_w, fwd_w, right_w, up_w, _ = compute_torso_frame(post_out, layout=layout)

                    # Anchor at pelvis/root. For COCO-17, change root index accordingly.
                    root_idx = 0  # H36M pelvis
                    origin = post_out[root_idx]  # World space origin

                    # Triad length relative to the scene size
                    radius = max(1e-6, (post_out.max(axis=0) - post_out.min(axis=0)).max() / 2.0)
                    s = float(getattr(args, "orientation_overlay_scale", 1.0)) * radius * 0.2

                    # Draw F (red), R (green), U (blue) in WORLD Space
                    ax.plot([origin[0], origin[0] + s * fwd_w[0]],
                            [origin[1], origin[1] + s * fwd_w[1]],
                            [origin[2], origin[2] + s * fwd_w[2]], lw=2, c='r')
                    ax.plot([origin[0], origin[0] + s * right_w[0]],
                            [origin[1], origin[1] + s * right_w[1]],
                            [origin[2], origin[2] + s * right_w[2]], lw=2, c='g')
                    ax.plot([origin[0], origin[0] + s * up_w[0]],
                            [origin[1], origin[1] + s * up_w[1]],
                            [origin[2], origin[2] + s * up_w[2]], lw=2, c='b')

                    # add legend once per figure
                    ax.plot([], [], c='r', label='Forward (Z)')
                    ax.plot([], [], c='g', label='Right (X)')
                    ax.plot([], [], c='b', label='Up (Y)')
                    ax.legend(loc='upper right')

                # (Optional) store a tidy row for analysis (camera-frame orientation)
                orientation_rows.append({
                    "frame_index": int(i),
                    "quat_w": float(prev_quat[0]), "quat_x": float(prev_quat[1]),
                    "quat_y": float(prev_quat[2]), "quat_z": float(prev_quat[3]),
                    "confidence": conf,
                    # Match CSV fieldnames exactly:
                    "forward_x": float(fwd_cam[0]), "forward_y": float(fwd_cam[1]), "forward_z": float(fwd_cam[2]),
                    "right_x":   float(right_cam[0]), "right_y":   float(right_cam[1]), "right_z":   float(right_cam[2]),
                    "up_x":      float(up_cam[0]),    "up_y":      float(up_cam[1]),    "up_z":      float(up_cam[2]),
                })

                # ----------------------------
                # Metric translation (depth → pos_xyz_m in camera frame)
                # ----------------------------
                pos_xyz_m = None
                trans_src = "none"
                trans_note = ""

                if getattr(args, "translation_source", "none") == "depth":
                    # --- Intrinsics & scale (defaults already on argparse) ---
                    fx, fy = float(args.fx), float(args.fy)
                    cx, cy = float(args.cx), float(args.cy)
                    depth_scale = float(getattr(args, "depth_scale", 0.001))

                    # --- Resolve the aligned depth PNG for THIS color frame ---
                    depth_path = None

                    # Case A: single-image run with --depth-path explicitly given
                    if is_image and getattr(args, "depth_path", None):
                        depth_path = args.depth_path

                    # Case B: "folder mode" (we call get_pose3D per image); infer filename from the color name
                    if depth_path is None and getattr(args, "depth_dir", None):
                        depth_dir = args.depth_dir
                        if depth_dir and not os.path.isabs(depth_dir) and not depth_dir.startswith('demo' + os.sep):
                            depth_dir = os.path.join('demo', 'image', depth_dir)
                        pattern = getattr(args, "depth_pattern", "depth_{index:04d}.png")
                        color_base = os.path.basename(video_path)

                        # Try to extract a trailing index (e.g., color_0141.png → 141)
                        m = re.search(r'(\d+)\.png$', color_base, flags=re.IGNORECASE)
                        if m:
                            idx = int(m.group(1))
                            cand = os.path.join(depth_dir, pattern.format(index=idx))
                            if os.path.exists(cand):
                                depth_path = cand

                        # Fallback 1: same basename inside depth_dir
                        if depth_path is None:
                            cand = os.path.join(depth_dir, color_base)
                            if os.path.exists(cand):
                                depth_path = cand

                        # Fallback 2: swap 'color'→'depth' in the basename
                        if depth_path is None:
                            swapped = re.sub(r'color', 'depth', color_base, flags=re.IGNORECASE)
                            cand = os.path.join(depth_dir, swapped)
                            if os.path.exists(cand):
                                depth_path = cand

                    # --- Read the depth image and compute translation ---
                    depth_u16 = _read_depth_png(depth_path)

                    # Root joint pixel from current overlay source (pelvis index = 0 in H36M)
                    root_idx = 0
                    if depth_u16 is None:
                        trans_src = "none"
                        trans_note = f"no_depth_image ({depth_path})"
                        _dbg("Metric translation (depth) SKIP", reason=trans_note)
                    elif overlay_2D_px is None or len(overlay_2D_px) <= root_idx:
                        trans_src = "none"
                        trans_note = "no_root_pixel"
                        _dbg("Metric translation (depth) SKIP", reason=trans_note)
                    else:
                        # Pixel to sample depth from (robust median in a small window)
                        u_px = float(overlay_2D_px[root_idx, 0])
                        v_px = float(overlay_2D_px[root_idx, 1])
                        Z_raw = _median_depth_at(depth_u16, u_px, v_px, ks=7)  # raw units (e.g., mm)
                        if (Z_raw is None) or (Z_raw <= 0):
                            trans_src = "none"
                            trans_note = "no_depth_at_root"
                            _dbg("Metric translation (depth) SKIP", reason=trans_note, u=u_px, v=v_px)
                        else:
                            Z_m = float(Z_raw) * depth_scale
                            X_m, Y_m, Z_m = _backproject_uvz_to_xyz(u_px, v_px, Z_m, fx, fy, cx, cy)
                            pos_xyz_m = [float(X_m), float(Y_m), float(Z_m)]
                            trans_src = "depth"
                            trans_note = f"depth_ok z={Z_m:.3f}m at (u={int(round(u_px))}, v={int(round(v_px))}) from {os.path.basename(depth_path)}"

                            # Detailed one-liner for your skip/diag reading:
                            _dbg("Metric translation (depth) OK",
                                 depth_path=depth_path, fx=fx, fy=fy, cx=cx, cy=cy,
                                 u=int(round(u_px)), v=int(round(v_px)),
                                 Z_raw=int(Z_raw), Z_m=float(Z_m),
                                 pos_xyz_m=pos_xyz_m)

                # -- Stage "final" PoseStamped for this frame (uses last_quat_for_log if available)
                # We'll commit the single best line once per frame after this block.
                t_ms = pos_ms if (not is_image and fps > 0) else 0.0
                # tx_ty_tz = post_out[0] if post_out is not None else np.zeros(3, dtype=np.float32)
                # quat = last_quat_for_log if last_quat_for_log is not None else np.zeros(4, dtype=np.float32)
                frame_id = getattr(args, "ros_frame_id", "camera_color_optical_frame")
                conf_min = float(getattr(args, "orientation_conf_min", 0.5))

                # Orientation status
                if getattr(args, "estimate_orientation", False) and ('conf' in locals()):
                    if conf < conf_min:
                        ori_ok = False
                        ori_status = f"SKIP:low_confidence(thresh={conf_min:.2f})"
                    else:
                        ori_ok = True
                        ori_status = "OK"
                else:
                    ori_ok = False
                    ori_status = "SKIP:orientation_disabled"

                # Translation status
                if getattr(args, "translation_source", "none") == "depth":
                    if pos_xyz_m is None:
                        trans_ok = False
                        trans_status = "SKIP:no_depth"
                    else:
                        trans_ok = True
                        trans_status = "OK"
                else:
                    trans_ok = (getattr(args, "translation_source", "none") == "none")
                    trans_status = "OK" if trans_ok else "SKIP:translation_unsupported"

                # Final status: OK only if orientation OK and (translation OK or translation=='none')
                if ori_ok and trans_ok:
                    final_status = "OK"
                elif not ori_ok:
                    final_status = ori_status
                else:
                    final_status = trans_status

                # Stage the final decision for this frame
                best_line = _stage_best(best_line, dict(
                frame_i = i, has_person = True, t_ms = t_ms,
                frame_id = frame_id,
                conf = (conf if 'conf' in locals() else None),
                yaw_deg = (yaw_deg if 'yaw_deg' in locals() else None),
                quat_wxyz = (prev_quat if 'prev_quat' in locals() else None),
                status = final_status,
                pos_xyz_m = pos_xyz_m,  # meters (or NA)
                translation_source = trans_src,  # "depth" or "none"
                note = (trans_note or "ok")))

            # ---- Commit one PoseStamped line per frame ----
            if pose_log_fp is not None and best_line is not None:
                _log_pose_stamped(
                    pose_log_fp,
                    frame_i = best_line["frame_i"],
                    has_person = best_line["has_person"],
                    t_ms = best_line["t_ms"],
                    frame_id = best_line["frame_id"],
                    conf = best_line["conf"],
                    yaw_deg = best_line["yaw_deg"],
                    quat_wxyz = best_line["quat_wxyz"],
                    status = best_line["status"],
                    pos_xyz_m = best_line["pos_xyz_m"],
                    translation_source = best_line["translation_source"],
                    note = best_line["note"])

            # (Optional) Single ROS publish per frame based on best_line
            if best_line is not None and best_line["status"] == "OK":
                # Build PoseStamped from best_line and publish once here.
                # ros_pub.publish(pose_msg)
                pass


            # -- Save 3D figure
            plt.savefig(output_dir_3D + str(('%04d' % i)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
            plt.clf()
            plt.close(fig)

    finally:
        # Always close resources
        if pose_log_fp is not None:
            pose_log_fp.close()
            print(f"[pose_log] Wrote: {pose_log_path}")
        try:
            if not is_image and cap is not None:
                cap.release()
        except Exception:
            pass


    print('Generating 3D pose successful!')

    # ----------------------------
    # 7) Optional: save orientation table (CSV/JSON)
    # ----------------------------

    if getattr(args, "estimate_orientation", False) and getattr(args, "orientation_save", None):
        out_path = Path(args.orientation_save)
        # If user gave only a filename (no directory), put it in this image's output_dir
        if not out_path.is_absolute() and out_path.parent == Path('.'):
            out_path = Path(output_dir) / out_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        if ext == ".csv":
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_index",
                        "quat_w", "quat_x", "quat_y", "quat_z",
                        "confidence",
                        "forward_x", "forward_y", "forward_z",
                        "right_x", "right_y", "right_z",
                        "up_x", "up_y", "up_z",
                    ]
                )
                writer.writeheader()
                writer.writerows(orientation_rows)
            print(f"[orientation] Saved CSV: {out_path}")
        else:
            p = out_path if ext == ".json" else out_path.with_suffix(".json")
            with p.open("w", encoding="utf-8") as f:
                json.dump(orientation_rows, f, ensure_ascii=False)
            print(f"[orientation] Saved JSON: {p}")

    # ----------------------------
    # 8) Compose side-by-side frames (2D | 3D)
    # ----------------------------

    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    # --- Skip compose if no 3D frames were generated ---
    if len(image_3d_dir) == 0:
        print("[WARN] No 3D frames generated — skipping compose step.")
        return

    if POSE_DEBUG:
        print(f"[COMPOSE.DBG] pose2D_frames={len(image_2d_dir)} pose3D_frames={len(image_3d_dir)}")

    print("\nGenerating demo...")
    output_dir_pose = os.path.join(output_dir, "pose/")
    os.makedirs(output_dir_pose, exist_ok=True)

    for i in tqdm(range(min(len(image_2d_dir), len(image_3d_dir)))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        # ## crop
        # edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        # image_2d = image_2d[:, edge:image_2d.shape[1] - edge]
        #
        # edge = 130
        # image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize=font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize=font_size)

        ## save
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose.png', dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
        if (i % 50) == 0:
            plt.close('all')  # just in case

    if POSE_DEBUG:
        print(f"[COMPOSE.DBG] wrote {len(image_2d_dir)} composite frames to {output_dir_pose}")


if __name__ == "__main__":
    _dbg("Raw sys.argv at start", argv=sys.argv)

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str, help='input video')
    group.add_argument('--image', type=str, help='input image or absolute path')
    group.add_argument('--image-dir', type=str, help='directory containing images')
    parser.add_argument('--gpu', type=str, default='0', help='Gpu id')
    # --- Orientation (all no-ops unless estimate is enabled) ---
    parser.add_argument('--estimate-orientation', action='store_true',
                        help='Compute torso orientation per frame (camera-frame).') #These flags are ignored unless --estimate-orientation is set
    parser.add_argument('--orientation-layout', type=str, default='h36m', choices=['h36m', 'coco17'],
                        help='Joint layout used to compute torso frame. Default: h36m.')
    parser.add_argument('--orientation-alpha', type=float, default=0.6,
                        help='SLERP-EMA smoothing factor ∈ [0,1], higher = smoother.')
    parser.add_argument('--orientation-conf-min', type=float, default=0.5,
                        help='If confidence below this, hold previous quat.')
    parser.add_argument('--orientation-overlay', action='store_true',
                        help='Draw orientation triad on 3D view.')
    parser.add_argument('--orientation-overlay-scale', type=float, default=1.0,
                        help='Scale of triad relative to scene size.')
    parser.add_argument('--orientation-save', type=str, default=None,
                        help='Path to save per-frame orientations (.csv or .json).')
    # --- PoseStamped logging (off by default; does not change current visuals) ---
    parser.add_argument('--log-pose-stamped', action='store_true',
                        help='If set, write a PoseStamped-style log per processed frame.')
    parser.add_argument('--pose-log-file', type=str, default=None,
                        help='Optional path for the PoseStamped log file. '
                             'Default: <output_dir>/pose_stamped.log')
    # --- ROS-shaped logging & publish toggles (plumbing only; safe on Windows) ---
    parser.add_argument('--ros-log-file', type=str, default=None,
                        help='PoseStamped-like log path (default: <output_dir>/pose_stamped.log).')
    parser.add_argument('--ros-publish', action='store_true',
                        help='Also publish PoseStamped (only on ROS machines). Ignored on Windows.')
    parser.add_argument('--ros-frame-id', type=str, default='camera_color_optical_frame',
                        help='Frame ID used in log/ROS PoseStamped (default: camera_color_optical_frame).')
    parser.add_argument('--translation-source', type=str, default='none',
                        choices=['none', 'depth', 'multiview', 'external'],
                        help='Metric translation source. Default: none (orientation-only logging later).')

    #This split between log-file and ros-log-file was intentional in your plan/execution to keep a dead-simple toggle for Windows runs and a
    # ROS-friendly path control for teammates.(so have to provide path for ros toggle.

    # --- Depth → metric translation (aligned 16UC1 PNGs in mm) ---
    parser.add_argument('--depth-path', type=str, default=None,
                        help='[image mode] Path to the aligned depth PNG for the given --image.')
    parser.add_argument('--depth-dir', type=str, default=None,
                        help='[video mode] Directory containing aligned depth PNGs.')
    parser.add_argument('--depth-pattern', type=str, default='depth_{index:04d}.png',
                        help='[video mode] Filename pattern for depth frames, e.g., depth_{index:04d}.png')
    parser.add_argument('--depth-scale', type=float, default=0.001,
                        help='Multiply raw depth units to convert to meters (mm→m = 0.001).')
    # Camera intrinsics (fx, fy, cx, cy) for back-projection
    parser.add_argument('--fx', type=float, default=911.47)
    parser.add_argument('--fy', type=float, default=911.56)
    parser.add_argument('--cx', type=float, default=654.27)
    parser.add_argument('--cy', type=float, default=366.90)

    parser.add_argument('--stale-timeout-ms', type=int, default=1000,
                        help='Max age (ms) to keep re-publishing the last-known pose when current frame is invalid.')
    parser.add_argument('--ros-topic', type=str, default='/human_pose',
                        help='ROS topic for PoseStamped.')
    parser.add_argument('--ros-diag-topic', type=str, default='/human_pose_status',
                        help='ROS topic for diagnostics (String JSON).')



    args = parser.parse_args()
    _dbg("Parsed args", video=args.video, image=args.image, gpu=args.gpu)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _dbg("CUDA_VISIBLE_DEVICES set", value=os.environ.get("CUDA_VISIBLE_DEVICES"))


    def _resolve_image_path(p: str) -> str:
        # absolute stays; otherwise prefix to demo/image/
        return p if os.path.isabs(p) else os.path.join('./demo/image/', p)


    def _list_images_in_dir(d: str):
        p = Path(d) if os.path.isabs(d) else Path('./demo/image') / d
        # common extensions
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        files = []
        for e in exts:
            files.extend(sorted(p.glob(e)))
        return [str(f) for f in files]


    if args.video:
        # ---- VIDEO MODE (unchanged) ----
        source_path = args.video if os.path.isabs(args.video) else './demo/video/' + args.video
        is_image = False
        _dbg("Resolved source (video)", source_path=source_path, is_image=is_image)
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        output_dir = './demo/output/' + source_name + '/'
        _dbg("Output directory", output_dir=output_dir)

        get_pose2D(source_path, output_dir)
        get_pose3D(source_path, output_dir, is_image=is_image, args=args)
        img2video(source_path, output_dir)
        print('Generating demo successful!')

    elif args.image:
        # ---- SINGLE IMAGE MODE (unchanged) ----
        source_path = _resolve_image_path(args.image)
        is_image = True
        _dbg("Resolved source (single image)", source_path=source_path, is_image=is_image)

        source_name = os.path.splitext(os.path.basename(source_path))[0]
        output_dir = './demo/output/' + source_name + '/'
        _dbg("Output directory", output_dir=output_dir)

        get_pose2D(source_path, output_dir)
        get_pose3D(source_path, output_dir, is_image=is_image, args=args)

        one_frame = os.path.join(output_dir, 'pose', '0000_pose.png')
        final_png = os.path.join(output_dir, f'{source_name}.png')
        if os.path.exists(one_frame):
            shutil.copyfile(one_frame, final_png)
            print(f'  ↳ Saved: {final_png}')
        else:
            print(f"[WARN] No composed pose image for {source_name}. "
                  f"Likely no person detected. Check {os.path.join(output_dir, 'skip.log')} if present.")
        print(f'Saved final image: {final_png}')
        print('Generating demo successful!')

    else:
        # ---- FOLDER MODE (new) ----
        images = _list_images_in_dir(args.image_dir)
        if not images:
            print(f'No images found in folder: {args.image_dir}')
            sys.exit(1)

        print(f'Found {len(images)} images in folder: {args.image_dir}')
        for idx, source_path in enumerate(images, 1):
            is_image = True
            source_name = os.path.splitext(os.path.basename(source_path))[0]
            output_dir = './demo/output/' + source_name + '/'
            print(f'[{idx}/{len(images)}] Processing: {source_name}')

            get_pose2D(source_path, output_dir)
            get_pose3D(source_path, output_dir, is_image=is_image, args=args)

            one_frame = os.path.join(output_dir, 'pose', '0000_pose.png')
            final_png = os.path.join(output_dir, f'{source_name}.png')
            if os.path.exists(one_frame):
                shutil.copyfile(one_frame, final_png)
                print(f'  ↳ Saved: {final_png}')
            else:
                print(f"[WARN] No composed pose image for {source_name}. "
                      f"Likely no person detected. Check {os.path.join(output_dir, 'skip.log')} if present.")

        print('Folder processing complete!')


#always have to be one folder above demo.
#Command: image-dir, image, video
# python demo/vis.py --video output_video.mp4 --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0
# python demo/vis.py --video output_video.mp4 --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0 --log-pose-stamped
# python demo/vis.py --image frame0141.jpg --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0 --ros-log-file ./demo/output/frame0141/pose_stamped.log

# python demo/vis.py --image new_test/color_0480.png --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0 --log-pose-stamped --translation-source depth --depth-path new_test/depth/depth_0480.png --depth-scale 0.001 --fx 911.47 --fy 911.56 --cx 654.27 --cy 366.90
#For video, use --depth-dir and --depth-pattern instead of --depth-path.


# python demo/vis.py --image-dir new_test/color --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0 --translation-source depth --depth-dir new_test/depth --depth-scale 0.001 --fx 911.47 --fy 911.56 --cx 654.27 --cy 366.90 --log-pose-stamped --orientation-save orientation.csv