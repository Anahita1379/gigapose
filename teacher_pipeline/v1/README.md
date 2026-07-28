# Teacher V1 baseline

This namespace preserves the tested compatibility pipeline implemented at the
top level of `teacher_pipeline`. Existing commands remain valid. Every command
also has an equivalent namespaced form, for example:

```bash
python3 -m teacher_pipeline.v1.build_observations --help
python3 -m teacher_pipeline.v1.refine_trajectories --help
python3 -m teacher_pipeline.v1.evaluate --help
```

The modules are intentionally thin aliases so fixes to the baseline have one
source of truth. See `teacher_pipeline/README.md` for its full contract.

