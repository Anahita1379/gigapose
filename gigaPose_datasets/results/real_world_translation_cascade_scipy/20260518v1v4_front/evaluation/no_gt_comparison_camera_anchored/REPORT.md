# GT-free tracking comparison

> These are disagreement and temporal-consistency measurements, not pose-accuracy errors.

Reference prediction model: `gigapose`.

## Coverage

| Model | Matched | Reference coverage | Model coverage |
| --- | ---: | ---: | ---: |
| selector_camera_anchored | 212 | 99.1% | 99.1% |
| translation_cascade_camera_anchored | 213 | 99.5% | 99.5% |
| scipy_graph_camera_anchored | 212 | 99.1% | 99.1% |

## Pose changes from the reference

| Model | Translation median (m) | Translation p90 (m) | Rotation median (deg) | Rotation p90 (deg) | Depth median (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| scipy_graph_camera_anchored | 3.207 | 9.312 | 17.314 | 50.221 | 3.050 |
| selector_camera_anchored | 3.063 | 9.550 | 17.290 | 50.468 | 2.881 |
| translation_cascade_camera_anchored | 2.960 | 9.804 | 17.326 | 50.319 | 2.811 |

A smaller change means closer agreement with the reference, not necessarily a more accurate pose. Review CAD overlays or an independent label source before deciding which model is better.

Confidence values from GigaPose and the tracker may have different calibration. Threshold plots are conditional summaries within each model and should not be interpreted as equal-probability operating points.
