# Exact reconstructed protocol

## Provenance and scope

Source: UCI 573, South German Credit, DOI 10.24432/C5QG88, CC BY 4.0.
Verified landing page: https://archive.ics.uci.edu/dataset/573/south+german+credit
Downloaded source archive URL:
https://archive.ics.uci.edu/static/public/573/south%2Bgerman%2Bcredit%2Bupdate.zip
Acquisition date: 2026-10-09. The raw data, source R file, code table and metadata
are bundled. The frozen split contains the source-file SHA-256, which is checked
before fitting. Row IDs are zero-based physical data-row positions, header excluded.

`kredit` / UCI `credit_risk` is 0=bad, 1=good. Only the explicitly named
`bad_credit` target uses 1=bad. The loader never infers label meaning from counts.
It validates the schema, integer fields, positive amount/duration, absence of
missing values and the 1,000/300 benchmark counts.

The monetary amount underwent an unknown monotonic transformation. Costs are
normalized hypothetical scenario quantities, not observed EAD or cash returns.
The historical sample oversamples bad credits; calibration describes this selected
benchmark and does not establish population default probabilities.

## Partition and preprocessing

- Split seed 2026: stratified 800 remaining, 200 test.
- Split seed 2027: remaining 800 -> 550 fitting, 250 remaining.
- Split seed 2028: remaining 250 -> 150 calibration, 100 policy.
- Class counts: fitting 165 bad, calibration 45, policy 30, test 60.
- Exact indices and their order are stored in `config/frozen_split.json`.
- StandardScaler on amount and duration, one-hot on 15 coded fields.
- Transformation fits once on all 550 original fitting records; unknown category
  levels subsequently receive all-zero one-hot groups. No calibration/policy/test
  row fits preprocessing. IDs and excluded fields never enter X.
- Each seed 11,22,33,44,55 draws 550 fitting indices with replacement using NumPy
  default_rng. The same draw is paired across all nine configurations.

These are frozen reconstructed partitions, not recovered manuscript partitions.

## Models, calibration and weights

All parameters are in `config/protocol.json`, including RNG seeds and otherwise
unspecified reconstruction choices. LR C=1, max_iter=2000, lbfgs, tolerance 1e-4.
RF 120 trees, minimum leaf 8; GB 100 depth-2 trees, learning rate .05, minimum
leaf 10. Other estimator defaults come from pinned scikit-learn 1.8.0.
Balanced weights give equal total class weight within each bootstrap. IDCS uses
Y*L+(1-Y)*margin, normalized to mean one **within that bootstrap**.

Unweighted, intercept-bearing logistic calibration has C=1, lbfgs, max_iter=2000,
tolerance 1e-4; it fits only on the 150 calibration rows. The manuscript does not
identify its regularization. C=1 is a documented reconstruction choice, retained
from the earlier ZIP. LR/GB raw scores are margins; RF raw score is P(bad).

The amount denominator is the median of the original 550 fitting rows. Margin is
e*(.06+.003*min(duration,60)); bad-credit loss is .60*e+.05. U minimizes realized
scenario loss on the 100 policy records over grid 0,.005,...,1; equal minima favor
the smallest threshold. E uses margin/(margin+loss). Approval is strictly p<tau;
equality rejects. Reject loss is zero; an approved good credit has negative margin
loss and an approved bad credit has positive bad-credit loss.

## SHAP and reason stability

LR attributions are exact linear interventional SHAP using **all original fitting
means**, avoiding any masker subsampling. TreeSHAP uses tree_path_dependent,
with no external background data. RF's raw TreeSHAP output is P(bad), not a
logit; GB's output is its raw margin. Affine calibration slope multiplies all
attributions and the intercept adjusts the base. This yields attributions of the
**calibrated logit**. Group one-hot dimensions by the original 17 variables.
Check raw and grouped/calibrated additivity at tolerance 1e-6 for every fitted model.

Select at most three strictly positive grouped values, largest first; ties use
alphabetical feature name. Any pair with an empty reason set receives undefined
Jaccard (NaN), even if only one set is empty. Means omit undefined values and
their denominators are explicitly reported. This operationalizes the manuscript's
empty-set rule conservatively; its original handling of a single empty set is unknown.

Ten seed pairs per configuration x 200 cases = 2,000 pairs. Report pooled means
over equal-action pairs and joint-denial pairs under E, and flips over all pairs.
Pairs are dependent observations, not 2,000 independent individuals. Each regime
can define different jointly denied subsets. Cross-family SHAP backgrounds differ.
Additivity is an implementation check, not causal, legal or surrogate validation.

## LIME

Use standard RF seed 11, the first paired bootstrap. Twenty fixed test positions
are drawn without replacement with RNG 2026, sorted in test-order, then stored by
row ID. Explainer seeds are 101,202,303, one explainer per seed. Each processes the
same ordered records, with 1,500 neighborhood samples and all 17 features.

LimeTabularExplainer uses classification mode, label 1=bad, no discretization of
continuous amount/duration, the 15 categorical dimensions sampled as categorical
codes, feature_selection='none', sample_around_instance=False; remaining settings
are lime 0.2.0.1 defaults. Its background is the original fitting partition and
prediction is calibrated P(bad). Signed positive coefficient lists use the same
top-3 and empty-set rules. LIME probability coefficients and SHAP logit contributions
are different objects; overlap is descriptive. R2 is weighted in-neighborhood fit,
not held-out fidelity. Marginal sampling can violate joint dependencies.

The original subset, explainer seeds and some LIME settings were absent from the
paper and previous ZIP. These choices are new and must be stated if adopted.

## Review and age audit

Mean five standard-RF calibrated probabilities, then apply E once. This differs
from averaging five per-seed losses. Uncertainty=(average rank of -abs(p-tau)-1)/(n-1);
closer gets higher score. Instability=1-mean all-pair reason Jaccard per record,
including both equal and unequal actions, without using labels. Undefined instability
is replaced with .5 **only for routing**, retained as missing in descriptive outputs
and flagged. Combined=.5*uncertainty+.5*instability. Both instability handling and
rank conventions are reconstruction choices, not recovered original settings.

Choose exactly 10/20/40 of the 200 cases, highest score first, ties by smaller
row ID. Labels never enter routing; they enter retrospective evaluation. Random
selection is integrated with inclusion probability k/n. Review scenarios (c,r):
(0,0),(.5,.05),(1,0), with cost .01 per reviewed record, following Equations 9-11.
The oracle is a bound; no claim of observed reviewer performance is made. For (0,0),
mean loss must rise by .01*budget for every strategy.

Sensitivity changes loss coefficient {.3,.6,.9} and margin multiplier {.5,1,1.5},
holding model and calibrated probabilities fixed. Age groups are <25 and >=25.
BA is bad-credit approval fraction among bad cases; GR is good-credit rejection
fraction among good cases. All denominators/counts are stored. No fairness or legal
conclusion follows from this small descriptive audit.

## Statistics and legal documentation

Table 1 sample loss SD uses ddof=1 over five training seeds. Table 2 averages
each record's loss across seeds, subtracts LR-standard-U for the same record, then
resamples those 200 paired differences 2,000 times with RNG 2026. Intervals are
2.5/97.5 percentile bounds, conditional on this split and model choices, without
multiplicity correction or population-shift uncertainty.

Technical dossiers retain p,tau,margin,loss,g,calibrated logit, log(L),-log(margin),
and D=logit(p)+log(L)-log(margin), with sign checked against E. Economic terms are
policy determinants, not independent borrower-feature SHAP effects. The actual
feature values supporting each ranked risk factor are recorded. Legal applicability
and expert dossier review are explicitly unverified; Table 8's normative scores
are not algorithmically generated empirical results. The package does not extend
the coauthor's legal analysis or infer notification sufficiency.
