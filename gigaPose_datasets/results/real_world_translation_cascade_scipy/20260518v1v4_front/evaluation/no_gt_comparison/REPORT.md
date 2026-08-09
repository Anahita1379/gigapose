# GT-free tracking comparison

> These are disagreement and temporal-consistency measurements, not pose-accuracy errors.

Reference prediction model: `gigapose`.

## Coverage

| Model | Matched | Reference coverage | Model coverage |
| --- | ---: | ---: | ---: |
| selector | 212 | 99.1% | 99.1% |
| translation_cascade | 212 | 99.1% | 99.1% |
| translation_cascade_scipy | 211 | 98.6% | 98.6% |

## Pose changes from the reference

| Model | Translation median (m) | Translation p90 (m) | Rotation median (deg) | Rotation p90 (deg) | Depth median (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| selector | 3.533 | 10.766 | 163.770 | 178.834 | 3.363 |
| translation_cascade | 3.512 | 10.301 | 163.770 | 178.834 | 3.433 |
| translation_cascade_scipy | 3.509 | 9.794 | 163.845 | 178.679 | 3.464 |

A smaller change means closer agreement with the reference, not necessarily a more accurate pose. Review CAD overlays or an independent label source before deciding which model is better.

Confidence values from GigaPose and the tracker may have different calibration. Threshold plots are conditional summaries within each model and should not be interpreted as equal-probability operating points.
