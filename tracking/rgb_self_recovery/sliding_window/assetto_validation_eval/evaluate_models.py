"""Run the shared evaluator with distinct raw/recovery overlay colors."""

from tracking.rgb_self_recovery.sliding_window import evaluate_backbones


evaluate_backbones.COLORS = [
    (235, 65, 55),
    (50, 110, 240),
    (245, 165, 35),
    (170, 65, 210),
    (20, 190, 190),
    (230, 90, 170),
    (125, 180, 45),
]


if __name__ == "__main__":
    evaluate_backbones.main()
