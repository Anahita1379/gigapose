# Pose-tracking paper draft

This directory contains a standalone, paper-style Methods and preliminary
Results draft for the RGB/CAD GRU selector, adaptive fixed-lag orientation, and
learned SE(3) filtering pipeline.

Build it from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_section.tex
```

The table currently reports the saved 328-frame Laguna-front/Putnam-rear
development-validation comparison. Replace or extend it after completing the
unseen 5,174-frame benchmark. Do not describe the current table as final test
performance.

The source is standalone for easy review. To integrate it into another paper,
copy the relevant sections, figure, macros, and bibliography entries into the
target conference template.
