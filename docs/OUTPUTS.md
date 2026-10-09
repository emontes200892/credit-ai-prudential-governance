# File dictionary

All tables use 1=bad credit. All costs are normalized scenario units. CSV `row_id`
always refers to the frozen source-file row, not a sorted DataFrame index.
Empty CSV numeric values indicate undefined values, not zero.

| File in results/reference_run | Meaning |
|---|---|
| manifest.json | Versions, source/split/protocol hashes, counts and verification status |
| protocol_used.json | Exact executed settings |
| split_indices.json | Exact four partitions and source digest |
| training_bootstrap_indices.csv | 550 ordered sampled row IDs per training seed |
| models/preprocessor.joblib | Transformation fitted on original fitting rows |
| models/{family}_{weighting}_{seed}.joblib | Estimator, calibrator, U threshold, data digest |
| metrics_by_seed.csv | 90 rows: 45 fits x 2 policies; predictive and economic metrics |
| test_predictions.csv | 18,000 rows: 90 configurations x 200 held-out cases |
| table1_performance.csv | Mean metrics; sample loss SD over training seeds |
| bootstrap_all_policy_comparisons.csv | Paired change in loss versus LR-standard-U, both policies |
| table2_loss_comparisons.csv | E-policy subset of paired conditional loss intervals |
| grouped_shap_values.csv | 153,000 grouped calibrated-logit attributions, 45 x 200 x 17 |
| additivity_checks.csv | Raw and grouped/calibrated reconstruction errors; calibration coefficients |
| decision_dossiers.csv | 9,000 technical case dossiers, one per model/case; not legal decisions |
| stability_pairs.csv | 18,000 record/seed-pair overlap and action records |
| stability_by_record.csv | Per-case all-pair overlap, instability, flips and valid-pair count |
| table3_explanation_stability.csv | Equal-action, joint-denial overlaps, flips and denominators |
| lime_record_ids.json | Twenty fixed held-out records used in LIME |
| lime_explanations.csv | Sixty explanations with neighborhood R2, reasons and overlap |
| lime_coefficients.csv | All 17 signed LIME coefficients per explanation |
| lime_seed_overlap.csv | Three explainer-seed pairs per LIME case |
| lime_summary.json | Mean neighborhood R2, seed overlap, SHAP/LIME overlap |
| rf_standard_ensemble_predictions.csv | Mean calibrated RF-standard probabilities, E decisions/loss |
| review_routing.csv | Label-free scores and exact inclusion weights for each budget/strategy |
| review_case_accounting.csv | Per-case retrospective loss/beneﬁt identities for all scenarios |
| review_all_scenarios.csv | 36 strategy/budget/scenario expected-loss comparisons |
| table4_imperfect_review.csv | Twelve comparisons for c=.5,r=.05,h=.01 |
| table5_cost_sensitivity.csv | Nine policy-only scenarios on fixed RF-standard ensemble |
| table6_age_audit.csv | Counts and rates for the two age groups |
| table*.tex | Generated plain LaTeX tabular fragments; wide tables may need scaling |
| figures/figure1_cost_stability.png/.pdf | Current-run configuration scatter, not a universal frontier |
| figures/figure2_imperfect_review.png/.pdf | Current-run hypothetical review curves |
| manuscript_comparison.csv | 203 reported-versus-reconstructed numeric comparisons |
| RUN_REPORT.md | Executed-run summary and numerical-replication limit |

Other package files:

- `manuscript/reported_table*.csv`: manual transcription of rounded PDF tables for
  comparison only. They never supply training labels, probabilities or figure points.
- `data/raw/`: original UCI bytes and code dictionary, plus repository metadata.
- `data/processed/`: English-named canonical CSV with explicitly recoded target.
- `config/frozen_split.json`: authoritative split for this reconstructed protocol.
- `notebooks/Explorar_reproducibilidad.ipynb`: executed results explorer, explanation
  guide, optional rerun cell and complete path handling.
- `CHECKSUMS.sha256`: integrity list for all bundled files except itself.
- `verify_package.py`: offline SHA-256 integrity verifier.

Expected loss in `test_predictions.csv` is the model's p-weighted loss for the
chosen action; scenario_loss uses the held-out label. Review expected loss integrates
hypothetical reviewer errors conditional on the actual retrospective label. These
three quantities must not be described as observed monetary performance.
