from tracking.rgb_self_recovery.sliding_window.assetto_export import (
    balanced_consecutive_subset,
    normalize_manifest,
    select_bins,
)


def _item(bin_name, run, frame, camera="front"):
    return {
        "bin_name": bin_name,
        "key": (run, frame, camera),
    }


def test_balanced_subset_uses_consecutive_run_blocks():
    index = [
        *[_item("distance0_20", "run_a", frame) for frame in range(20)],
        *[_item("distance0_20", "run_b", frame) for frame in range(100, 120)],
    ]
    selected, report = balanced_consecutive_subset(index, 10, "error")
    by_run = {
        run: [item["key"][1] for item in selected if item["key"][0] == run]
        for run in ("run_a", "run_b")
    }
    assert len(selected) == 10
    assert all(len(frames) == 5 for frames in by_run.values())
    assert all(
        frames == list(range(frames[0], frames[0] + len(frames)))
        for frames in by_run.values()
    )
    assert report["bins"]["distance0_20"]["selected"] == 10


def test_short_bin_all_policy_never_duplicates():
    index = [_item("distance140_plus", "run_a", frame) for frame in range(3)]
    selected, report = balanced_consecutive_subset(index, 529, "all")
    assert selected == index
    assert report["bins"]["distance140_plus"]["short"] is True


def test_select_bins_excludes_unrequested_far_ranges():
    index = [
        _item("distance100_120", "run_a", 1),
        _item("distance120_140", "run_a", 2),
        _item("distance140_plus", "run_a", 3),
    ]
    selected, names = select_bins(index, "distance100_120")
    assert names == ["distance100_120"]
    assert [item["bin_name"] for item in selected] == ["distance100_120"]


def test_normalize_manifest_accepts_benchmark_schema():
    manifest = {
        "source_run": "run_a",
        "frame": "42",
        "camera_id": "rear",
        "output_filename": "run_a__000042.jpg",
    }
    normalized = normalize_manifest(manifest, {"sim_time_ms": "1234"})
    assert normalized["source_frame"] == "42"
    assert normalized["sim_time_ms"] == "1234"
    assert normalized["image"] == "images/rear/run_a__000042.jpg"
    assert normalized["masks"] == "masks/rear/run_a__000042.png"


def test_normalize_manifest_preserves_legacy_paths():
    manifest = {
        "source_run": "run_a",
        "source_frame": "7",
        "camera_id": "front",
        "sim_time_ms": "900",
        "image": "legacy/image.jpg",
        "masks": "legacy/mask.png",
    }
    normalized = normalize_manifest(manifest, {"sim_time_ms": "901"})
    assert normalized["frame"] == "7"
    assert normalized["sim_time_ms"] == "900"
    assert normalized["image"] == "legacy/image.jpg"
    assert normalized["masks"] == "legacy/mask.png"
