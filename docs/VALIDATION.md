# Validation performed on 2026-10-09

Verified environment: Linux x86_64, CPython 3.12.14. Exact computational dependency
versions are recorded in requirements.lock.txt and results/reference_run/manifest.json.

## Completed checks

- Downloaded the current official UCI 573 archive; inspected its source code table.
  Confirmed kredit=0 bad / 1 good and explicitly recoded 300 bad / 700 good records.
- Validated the exact schema, no missing/nonfinite values, 17 model predictors,
  excluded fields, stratified class counts, four disjoint partitions and full coverage.
- Executed all 45 estimators, 90 model-policy evaluations, 18,000 held-out action
  records, 153,000 grouped SHAP values and 60 LIME explanations.
- Raw-score and calibrated/grouped SHAP additive reconstruction errors were below
  1e-6 for every model (observed maxima below 5e-15).
- Installed all locked computational dependencies from scratch in a new virtual
  environment, with no access to the previous environment's site-packages. pip check
  reported no broken requirements.
- Ran the entire protocol again in that clean environment. All 23 CSV output files
  agreed with the prior execution at rtol=1e-10, atol=1e-12, with identical schemas,
  record order and text values. No matching-to-paper tuning was performed.
- Verified that all 45 bundled serialized model/calibrator pairs reproduce their
  stored probabilities and actions on the same test records.
- Passed 15 automated accounting, data and integration tests. Tests cover target
  meaning, feature exclusions, equivalent canonical CSV, overlap detection,
  threshold ties, policy algebra, class/IDCS weights, conditional paired bootstrap,
  empty reasons, label-free routing, random-review integration, reviewer-benefit
  identity, oracle/no-improvement accounting, saved-model outputs, SHAP
  reconstruction, sensitivity-baseline agreement and age denominators.
- Executed and validated the 10 code cells of the 20-cell notebook without errors
  using sequential in-process IPython execution. The execution environment blocks
  socket binding, so a live Jupyter kernel/server could not be tested here. The
  included validate_notebook.py reproduces the in-process check and captures outputs.
- Visually inspected both final figures. Figure outputs originate from predictions
  and explanations of this run; no rounded manuscript data generate them.
- Preserved the supplied manuscript unchanged and generated 203 numerical
  comparisons. Only 4 agree within the manuscript's rounding precision. This is
  evidence that the reconstructed run does not replicate the original tables,
  not evidence that either split is the original one.
- Prepared SHA-256 checksums of all packaged files and verified the extracted ZIP.

## Limits

The tests demonstrate consistency and repeatability of this reconstructed
implementation in the verified environment. They do not recover the original split,
validate the original reported values, prove causal explanation faithfulness,
validate reviewers or legal notices, or establish external financial performance.
Windows/macOS and other Python versions were not independently executed. Dependency
installation requires internet; the core experiment and integrity check run offline.

Run the provided tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Run the protocol with:

```bash
python run_experiments.py --out results/my_run
python compare_runs.py results/reference_run results/my_run
```
