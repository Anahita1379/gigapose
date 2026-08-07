"""CLI for optional soft-map, five-frame RGB self-recovery."""

from pathlib import Path

from tracking.rgb_self_recovery.sliding_window.runner import build_parser, run_tracker

from .map_selector import MapWindowSelector


def parser():
    p = build_parser("Map-aware five-frame RGB self-recovery")
    p.add_argument("--track-map", type=Path)
    p.add_argument("--map-extrinsics", type=Path)
    p.add_argument("--map-metadata-root", type=Path)
    p.add_argument("--map-transform-unit", choices=("m", "mm"), default="m")
    p.add_argument("--map-weight", type=float, default=0., help="0 records diagnostics only")
    p.add_argument("--map-motion-weight", type=float, default=0., help="0 disables Frenet motion reranking")
    p.add_argument("--map-boundary-sigma-m", type=float, default=.5)
    p.add_argument("--map-height-sigma-m", type=float, default=.5)
    p.add_argument("--map-expected-center-height-m", type=float, default=0.)
    p.add_argument("--map-heading-sigma-deg", type=float, default=20)
    p.add_argument("--map-car-width-m", type=float, default=2.)
    p.add_argument("--map-forward-axis", choices=("x", "y", "z"), default="x")
    p.add_argument("--map-allow-reverse", action="store_true")
    p.add_argument("--map-speed-sigma-mps", type=float, default=5)
    p.add_argument("--map-acceleration-sigma-mps2", type=float, default=10)
    p.add_argument("--map-lateral-speed-sigma-mps", type=float, default=2)
    p.add_argument("--map-backward-speed-sigma-mps", type=float, default=2)
    p.add_argument("--map-hard-boundary-violation-m", type=float, default=float("inf"))
    return p


def main():
    args = parser().parse_args()
    selector = MapWindowSelector(args)
    run_tracker(args, selector, "rgb_self_recovery_map_window_v1")


if __name__ == "__main__":
    main()
