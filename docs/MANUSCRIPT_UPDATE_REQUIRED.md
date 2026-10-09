# Before claiming numerical reproducibility of the manuscript

The supplied PDF says original split indices and data digest were not included.
This package supplies frozen indices and a digest for a **new reconstruction**,
not the original experiment. The distinction must remain explicit.

## Two defensible paths

1. Recover the original data version, ordered split indices, preprocessing,
   calibration settings, trained models or code, and LIME subset/seeds. Execute
   that recovered protocol and verify the supplied manuscript's numbers.
2. Adopt this verified reconstructed run. Update the abstract, Tables 1-6, Figures
   1-2, LIME summary and every numerical statement in Sections 6 and elsewhere
   from these outputs. Reassess all claims, directions and confidence intervals.
   Do not substitute tables without updating the narrative and conclusions.

This task produces the reproducibility package; the PDF has **not been revised**.
The complete cell-level discrepancy report is
`results/reference_run/manuscript_comparison.csv`. Agreement after rounding in
isolated cells does not certify recovered provenance.

## Methodological disclosures to add if adopting this package

- Cite the current downloaded South German Credit entry (UCI 573, 2020,
  DOI 10.24432/C5QG88), with its raw-file checksum. The earlier UCI 522/2019
  reference in the PDF does not identify this downloaded version uniquely.
- State that UCI's 0=bad,1=good target was transformed to Y=1-binary_target.
- Describe exact stratified split construction with seeds 2026,2027,2028 and
  retain all indices, including their order.
- State that preprocessing fits once on the original 550 fitting rows and is
  reused across paired fitting bootstraps.
- Record logistic-calibration C=1 and all pinned dependency versions.
- State fitting-set means for LR SHAP, tree-path reference for RF/GB and affine
  calibration of raw attributions into calibrated-logit attributions.
- State any empty-set convention, reason-ranking ties and pooled pair denominators.
- State LIME subset seed, row IDs, explainer seeds 101/202/303, output probability,
  no numeric discretization and all 17-feature local surrogate.
- Document uncertainty rank normalization, undefined-instability fill .5 for
  routing only, and routing ties by row ID.
- Record sample SD convention (ddof=1) and conditional bootstrap RNG 2026.

Suggested replacement for the missing-provenance sentence in Section 3.1,
**only if the paper is updated to this executed run**:

> The reproducibility package includes the original downloaded data, a SHA-256
> digest, exact ordered split indices, paired training-bootstrap indices,
> parameter settings, software versions, fitted estimators and calibrators,
> per-record outputs, and scripts generating the tables and figures. This
> version uses newly frozen reconstructed partitions rather than recovered
> partitions from the earlier pilot.

Legal rubric scores are the legal coauthor's normative assessments. They remain
separate from algorithmic outputs and are not recast as empirically validated.
No LendingClub/HMDA experiment, live-bank capital estimate or systemic-effect
estimate is added by this package.
