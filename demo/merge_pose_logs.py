#!/usr/bin/env python3
# merge_pose_logs.py
import argparse
import os
import re
from pathlib import Path

HEADER_TITLE = "# PoseStamped log (one line per frame)"
HEADER_FIELDS = "# fields: frame_index | has_person | t_ms | frame_id | conf | yaw_deg | qw qx qy qz | status | pos_xyz_m | translation_source | note"

def find_sequence_dirs(base_dir: Path):
    """Return subdirs that contain a pose_stamped.log, sorted naturally by any trailing number."""
    seq_dirs = []
    for p in sorted(base_dir.iterdir()):
        if p.is_dir() and (p / "pose_stamped.log").exists():
            seq_dirs.append(p)

    # Natural-ish sort by numeric tail if present (color_000123 -> 123), else lex
    def dir_key(p: Path):
        m = re.search(r'(\d+)$', p.name)
        return (p.name, int(m.group(1))) if m else (p.name, -1)

    seq_dirs.sort(key=dir_key)
    return seq_dirs

def parse_status(line: str):
    """
    Robustly parse the 'status' field (8th pipe-separated token, 0-indexed).
    Return (ok_bool, tokens_or_none). If it can't parse, returns (None, None).
    """
    s = line.strip()
    if not s or s.startswith('#'):
        return (None, None)

    parts = [t.strip() for t in line.split('|')]
    # Expect 11 fields:
    # 0:frame_index 1:has_person 2:t_ms 3:frame_id 4:conf 5:yaw_deg
    # 6:qw..qz 7:status 8:pos_xyz_m 9:translation_source 10:note
    if len(parts) < 11:
        return (None, None)

    status = parts[7]
    return (status == "OK", parts)

def rewrite_with_global_index(parts, new_idx: int) -> str:
    """Replace frame_index with zero-padded global index and reassemble line."""
    parts = parts[:]  # copy
    parts[0] = f"{new_idx:06d}"  # zero-padded 6 digits
    return " | ".join(parts)

def merge_logs(base_dir: Path, out_all: Path, out_ok: Path):
    seq_dirs = find_sequence_dirs(base_dir)
    if not seq_dirs:
        print(f"[ERR] No sequence folders with pose_stamped.log found under: {base_dir}")
        return 1

    print(f"[INFO] Found {len(seq_dirs)} sequences.")
    gidx_all = 0
    gidx_ok  = 0

    with out_all.open("w", encoding="utf-8") as fa, out_ok.open("w", encoding="utf-8") as fo:
        # Write a single header into each merged file
        fa.write(HEADER_TITLE + "\n")
        fa.write(HEADER_FIELDS + "\n")
        fo.write(HEADER_TITLE + "\n")
        fo.write(HEADER_FIELDS + "\n")

        for seq in seq_dirs:
            src = seq / "pose_stamped.log"
            # Comment label so humans know which block came from which folder; replay ignores lines starting '#'
            fa.write(f"# seq={seq.name}\n")
            fo.write(f"# seq={seq.name}\n")

            with src.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if s.startswith('#'):
                        # skip per-file headers/comments
                        continue

                    ok, parts = parse_status(line)
                    if parts is None:
                        # malformed line, keep it in 'all' to be safe, but don't break format
                        # try to coerce: just append line as-is (won’t be reindexed)
                        # safer approach: skip malformed entries
                        continue

                    # Reindex for ALL
                    fa.write(rewrite_with_global_index(parts, gidx_all) + "\n")
                    gidx_all += 1

                    # Reindex for OK-only
                    if ok:
                        fo.write(rewrite_with_global_index(parts, gidx_ok) + "\n")
                        gidx_ok += 1

    print(f"[DONE] Wrote:\n  - {out_all} (frames={gidx_all})\n  - {out_ok} (OK frames={gidx_ok})")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Merge PoseStamped logs from PoseFormerV2 outputs.")
    parser.add_argument("--base", required=True,
                        help="Base output folder containing color_******/pose_stamped.log")
    parser.add_argument("--out-all", default="pose_stamped_all.log",
                        help="Merged log (all lines) filename")
    parser.add_argument("--out-ok", default="pose_stamped_ok.log",
                        help="Merged log (OK-only) filename")
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser().resolve()
    out_all = (base_dir / args.out_all).resolve()
    out_ok  = (base_dir / args.out_ok).resolve()

    code = merge_logs(base_dir, out_all, out_ok)
    raise SystemExit(code)

if __name__ == "__main__":
    main()