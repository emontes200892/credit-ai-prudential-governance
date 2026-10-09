"""Integration checks against the genuinely executed bundled reference run."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from credit_ai.core import FEATURES, load_data, raw_score, scenario_loss

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/reference_run'


def test_expected_experiment_counts_and_additivity():
    manifest = json.loads((RUN/'manifest.json').read_text())
    assert manifest['fitted_models'] == 45
    assert len(pd.read_csv(RUN/'metrics_by_seed.csv')) == 90
    assert len(pd.read_csv(RUN/'test_predictions.csv')) == 18000
    assert len(pd.read_csv(RUN/'stability_pairs.csv')) == 18000
    assert len(pd.read_csv(RUN/'grouped_shap_values.csv')) == 153000
    assert len(pd.read_csv(RUN/'lime_explanations.csv')) == 60
    assert manifest['original_numerical_replication_certified'] is False
    checks = pd.read_csv(RUN/'additivity_checks.csv')
    assert checks.raw_additivity_max_error.max() < 1e-6
    assert checks.calibrated_grouped_additivity_max_error.max() < 1e-6


def test_all_saved_models_reproduce_prediction_records():
    x, y, _, _ = load_data(ROOT/'data/raw/SouthGermanCredit.asc')
    te = np.asarray(json.loads((ROOT/'config/frozen_split.json').read_text())['indices']['test'])
    prep = joblib.load(RUN/'models/preprocessor.joblib')
    z = prep.transform(x.iloc[te])
    pred = pd.read_csv(RUN/'test_predictions.csv')
    for f in sorted((RUN/'models').glob('*_*_*.joblib')):
        obj = joblib.load(f)
        raw = raw_score(obj['model'], z)
        p = obj['calibration'].predict_proba(raw.reshape(-1, 1))[:, 1]
        rows = pred[(pred.model == obj['family']) & (pred.weighting == obj['weighting']) &
                    (pred.seed == obj['seed']) & (pred.policy == 'E')].set_index('row_id').loc[te]
        np.testing.assert_allclose(p, rows.prob_bad, atol=1e-14, rtol=1e-13)
        np.testing.assert_array_equal(p < rows.threshold, rows.approve)
        np.testing.assert_allclose(scenario_loss(rows.approve, y[te], rows.margin, rows.loss_bad), rows.scenario_loss)


def test_grouped_attributions_reconstruct_calibrated_logit():
    values = pd.read_csv(RUN/'grouped_shap_values.csv')
    sums = values.groupby(['model', 'weighting', 'seed', 'row_id']).calibrated_logit_shap.sum().reset_index()
    checks = pd.read_csv(RUN/'additivity_checks.csv')
    dossiers = pd.read_csv(RUN/'decision_dossiers.csv')
    result = sums.merge(checks, on=['model','weighting','seed']).merge(dossiers, on=['model','weighting','seed','row_id'])
    np.testing.assert_allclose(result.calibrated_logit_shap + result.calibrated_logit_base,
                               result.calibrated_logit, atol=1e-6)


def test_ensemble_sensitivity_and_review_accounting():
    ens = pd.read_csv(RUN/'rf_standard_ensemble_predictions.csv')
    sens = pd.read_csv(RUN/'table5_cost_sensitivity.csv')
    base = sens[(sens.loss_coefficient == .6) & (sens.margin_multiplier == 1)].iloc[0]
    assert np.isclose(base.scenario_loss, ens.scenario_loss.mean())
    assert np.isclose(base.approval_pct, ens.approve.mean()*100)
    review = pd.read_csv(RUN/'review_all_scenarios.csv')
    none = review.query("scenario == 'no_improvement'")
    np.testing.assert_allclose(none.delta_loss, none.budget * .01)
    assert np.allclose(review.baseline_loss, ens.scenario_loss.mean())
    ages = pd.read_csv(RUN/'table6_age_audit.csv')
    assert ages.n.sum() == 200 and ages.bad_n.sum() == 60 and ages.good_n.sum() == 140
