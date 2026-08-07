"""Run five-frame recovery with a DINO patch matching U-Net checkpoint."""

from tracking.rgb_self_recovery.sliding_window import runner

from .inference import load_predictor


def main() -> None:
    # This process-local override leaves the shared runner source untouched.
    runner.load_predictor = load_predictor
    args = runner.build_parser(description=__doc__).parse_args()
    runner.run_tracker(
        args,
        report_format="rgb_self_recovery_sliding_window_dino_matching_unet_v1",
    )


if __name__ == "__main__":
    main()
