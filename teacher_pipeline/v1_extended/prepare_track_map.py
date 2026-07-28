"""Prepare a centerline/raceline for the physical V1 teacher."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .track_map import _unit


def prepare(center, left_width, right_width, closed=True):
    center = np.asarray(center, dtype=float)
    if closed and np.linalg.norm(center[0] - center[-1]) < 1e-6:
        center = center[:-1]
        left_width, right_width = left_width[:-1], right_width[:-1]
    previous = np.roll(center, 1, axis=0) if closed else np.vstack([center[0], center[:-1]])
    following = np.roll(center, -1, axis=0) if closed else np.vstack([center[1:], center[-1]])
    tangent = _unit(following - previous)
    up = np.tile([0.0, 0.0, 1.0], (len(center), 1))
    lateral = _unit(np.cross(up, tangent))
    normal = _unit(np.cross(tangent, lateral))
    segment = np.linalg.norm(np.diff(center, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(segment)]
    closing = float(np.linalg.norm(center[0] - center[-1])) if closed else 0.0
    return dict(
        center_xyz=center, s=s, tangent_xyz=tangent,
        lateral_xyz=lateral, normal_xyz=normal,
        left_width=np.asarray(left_width, dtype=float),
        right_width=np.asarray(right_width, dtype=float),
        closed=np.asarray(closed), track_length_m=np.asarray(s[-1] + closing),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centerline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-half-width-m", type=float, default=6.0)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.centerline.suffix.lower() == ".npz":
        source = np.load(args.centerline)
        center = source["center_xyz"]
        left = source["left_width"] if "left_width" in source else np.full(len(center), args.default_half_width_m)
        right = source["right_width"] if "right_width" in source else np.full(len(center), args.default_half_width_m)
    else:
        table = np.genfromtxt(args.centerline, delimiter=",", names=True)
        center = np.column_stack([table[name] for name in ("x", "y", "z")])
        names = table.dtype.names or ()
        left = table["left_width"] if "left_width" in names else np.full(len(center), args.default_half_width_m)
        right = table["right_width"] if "right_width" in names else np.full(len(center), args.default_half_width_m)
    output = prepare(center, left, right, not args.open)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"Wrote {len(center)} track samples ({float(output['track_length_m']):.1f} m) to {args.output}")


if __name__ == "__main__":
    main()
