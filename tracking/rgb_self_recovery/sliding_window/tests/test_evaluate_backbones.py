from tracking.rgb_self_recovery.sliding_window.evaluate_backbones import (
    _stats,
    add_error_m,
    paired_comparisons,
    summarize,
)


def _record(distance, cnn_t, cnn_r, dino_t, dino_r):
    return {
        "distance_m": distance,
        "models": {
            "cnn": {
                "translation_error_m": cnn_t,
                "rotation_error_deg": cnn_r,
            },
            "dino": {
                "translation_error_m": dino_t,
                "rotation_error_deg": dino_r,
            },
        },
    }


def test_paired_improvement_is_positive_when_dino_error_is_lower():
    records = [
        _record(10, 2.0, 20.0, 1.0, 10.0),
        _record(30, 4.0, 40.0, 3.0, 30.0),
    ]
    comparison = paired_comparisons(records, "cnn", ["cnn", "dino"])["dino"]
    assert comparison["translation_improvement_m"]["median"] == 1.0
    assert comparison["rotation_improvement_deg"]["median"] == 10.0
    assert comparison["both_better_fraction"] == 1.0


def test_summary_uses_ground_truth_distance_bins():
    records = [
        _record(10, 2.0, 20.0, 1.0, 10.0),
        _record(30, 4.0, 40.0, 3.0, 30.0),
    ]
    summary = summarize(records, ["cnn", "dino"], [0, 20, 40])
    assert summary["cnn"]["distance_bins"][0]["translation_error_m"]["count"] == 1
    assert summary["cnn"]["distance_bins"][1]["translation_error_m"]["count"] == 1


def test_stats_reports_mean_and_rmse():
    result = _stats([3.0, 4.0])
    assert result["mean"] == 3.5
    assert result["rmse"] == (12.5 ** 0.5)


def test_add_is_corresponding_vertex_error_in_metres():
    vertices = __import__("numpy").asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    prediction = __import__("numpy").eye(4)
    truth = __import__("numpy").eye(4)
    prediction[0, 3] = 2.0
    assert add_error_m(prediction, truth, vertices) == 2.0
