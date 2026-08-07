"""Run five-frame RGB self-recovery with a matching U-Net checkpoint."""

from tracking.rgb_self_recovery.sliding_window import runner

from .inference import load_predictor


def main() -> None:
    # The parent runner intentionally exposes its predictor loader as a module
    # dependency. Override it only in this isolated process; no source or
    # persistent global state outside this Python invocation is changed.
    runner.load_predictor = load_predictor
    args = runner.build_parser(description=__doc__).parse_args()
    runner.run_tracker(
        args,
        report_format="rgb_self_recovery_sliding_window_matching_unet_v1",
    )


if __name__ == "__main__":
    main()
