"""Run every implemented historical pilot experiment; never fabricate matches."""
import argparse
import importlib.metadata
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .analysis import age_audit, review_analysis, sensitivity_analysis
from .core import (FEATURES, digest, financial_terms, load_data, make_preprocessor,
                   make_split, model_expected_loss, paired_ci, positive_reasons,
                   raw_score, scenario_loss, select_threshold, training_weights,
                   validate_split, write_json)
from .explain import grouped_shap, lime_diagnostics, stability_rows
from .report import create_reports

ROOT = Path(__file__).resolve().parents[2]


def make_model(family, seed, cfg):
    cls = {"LR": LogisticRegression, "RF": RandomForestClassifier, "GB": GradientBoostingClassifier}[family]
    return cls(**cfg["model_parameters"][family], random_state=seed)


def run(data_path, config_path, split_path, out):
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}. Choose a fresh --out directory.")
    cfg = json.loads(config_path.read_text())
    x, y, age, full_data = load_data(data_path)
    split_payload = json.loads(split_path.read_text())
    if split_payload["source_sha256"] != digest(data_path):
        raise ValueError("Frozen partitions belong to another data file; checksum mismatch")
    split = {k: np.asarray(v, dtype=int) for k, v in split_payload["indices"].items()}
    validate_split(split, y)
    out.mkdir(parents=True)
    (out / "models").mkdir()
    write_json(out / "protocol_used.json", cfg)
    write_json(out / "split_indices.json", split_payload)
    fi, ca, po, te = [split[k] for k in ("fit", "calibration", "policy", "test")]
    exposure, margin, loss = financial_terms(x, fi, cfg)
    # Preprocessing is fitted once on the original 550 fitting rows, NOT re-fitted
    # on bootstraps. Calibration, policy and test never fit the transformation.
    prep = make_preprocessor().fit(x.iloc[fi])
    z = prep.transform(x)
    joblib.dump(prep, out / "models" / "preprocessor.joblib", compress=3)
    metrics, predictions, attributions, checks, dossiers, fit_samples = [], [], [], [], [], []
    fits = defaultdict(list)
    for seed in cfg["training_seeds"]:
        sample = np.random.default_rng(seed).choice(fi, size=len(fi), replace=True)
        fit_samples.extend({"seed": seed, "position": j, "row_id": int(i)} for j, i in enumerate(sample))
        for family in cfg["model_families"]:
            for regime in cfg["weighting_regimes"]:
                weights = training_weights(y[sample], margin[sample], loss[sample], regime)
                model = make_model(family, seed, cfg).fit(z[sample], y[sample], sample_weight=weights)
                calibration = LogisticRegression(**cfg["calibration"])
                calibration.fit(raw_score(model, z[ca]).reshape(-1, 1), y[ca])
                p_policy = calibration.predict_proba(raw_score(model, z[po]).reshape(-1, 1))[:, 1]
                raw_test = raw_score(model, z[te])
                p_test = calibration.predict_proba(raw_test.reshape(-1, 1))[:, 1]
                logit = calibration.decision_function(raw_test.reshape(-1, 1))
                uniform = select_threshold(p_policy, y[po], margin[po], loss[po],
                                           cfg["uniform_threshold_grid_step"])
                tau = margin[te] / (margin[te] + loss[te])
                grouped, base, raw_error, calibrated_error = grouped_shap(model, family, prep, calibration,
                                                                         z[fi], z[te])
                reasons = [positive_reasons(row) for row in grouped]
                key = {"seed": seed, "model": family, "weighting": regime}
                checks.append({**key, "raw_additivity_max_error": raw_error,
                               "calibrated_grouped_additivity_max_error": calibrated_error,
                               "calibration_slope": float(calibration.coef_[0, 0]),
                               "calibration_intercept": float(calibration.intercept_[0]),
                               "calibrated_logit_base": base})
                for pos, row_id in enumerate(te):
                    attributions.extend({**key, "row_id": int(row_id), "feature": feature,
                                         "calibrated_logit_shap": float(grouped[pos, j])}
                                        for j, feature in enumerate(FEATURES))
                    g = p_test[pos] * loss[row_id] - (1 - p_test[pos]) * margin[row_id]
                    policy_d = logit[pos] + np.log(loss[row_id]) - np.log(margin[row_id])
                    if bool(policy_d < 0) != bool(p_test[pos] < tau[pos]):
                        raise AssertionError("Policy decomposition sign disagrees with economic threshold")
                    dossiers.append({**key, "row_id": int(row_id), "truth": int(y[row_id]),
                                     "raw_score": float(raw_test[pos]), "calibrated_logit": float(logit[pos]),
                                     "prob_bad": float(p_test[pos]), "tau": float(tau[pos]),
                                     "margin": float(margin[row_id]), "loss_bad": float(loss[row_id]),
                                     "expected_approval_loss": float(g), "policy_D": float(policy_d),
                                     "log_loss_policy_term": float(np.log(loss[row_id])),
                                     "negative_log_margin_policy_term": float(-np.log(margin[row_id])),
                                     "positive_top3_risk_factors": "|".join(reasons[pos]),
                                     "economic_approve": bool(p_test[pos] < tau[pos]),
                                     "factor_values": json.dumps({f: int(x.iloc[row_id][f]) for f in reasons[pos]}),
                                     "jurisdiction_verified": False, "legal_review_status": "not_evaluated"})
                for policy in ("U", "E"):
                    threshold = np.full(len(te), uniform) if policy == "U" else tau
                    approve = p_test < threshold
                    realized = scenario_loss(approve, y[te], margin[te], loss[te])
                    expected = model_expected_loss(approve, p_test, margin[te], loss[te])
                    metrics.append({**key, "policy": policy, "auroc": roc_auc_score(y[te], p_test),
                                    "average_precision": average_precision_score(y[te], p_test),
                                    "brier": brier_score_loss(y[te], p_test),
                                    "approval_pct": float(approve.mean() * 100),
                                    "scenario_loss": float(realized.mean()),
                                    "model_expected_loss": float(expected.mean()), "uniform_threshold": uniform})
                    for pos, row_id in enumerate(te):
                        predictions.append({**key, "policy": policy, "row_id": int(row_id),
                                            "truth": int(y[row_id]), "prob_bad": p_test[pos],
                                            "threshold": threshold[pos], "approve": bool(approve[pos]),
                                            "scenario_loss": realized[pos], "model_expected_loss": expected[pos],
                                            "margin": margin[row_id], "loss_bad": loss[row_id]})
                entry = {"seed": seed, "model": model, "calibration": calibration,
                         "p_test": p_test, "reasons": reasons}
                fits[(family, regime)].append(entry)
                model_path = out / "models" / f"{family}_{regime}_{seed}.joblib"
                joblib.dump({"family": family, "weighting": regime, "seed": seed,
                             "model": model, "calibration": calibration, "uniform_threshold": uniform,
                             "source_sha256": digest(data_path)},
                            model_path, compress=3)
                stored = joblib.load(model_path)
                stored_p = stored['calibration'].predict_proba(raw_score(stored['model'], z[te]).reshape(-1, 1))[:, 1]
                if not np.allclose(stored_p, p_test, atol=1e-14, rtol=1e-13):
                    raise AssertionError("Serialized model does not reproduce its predictions")
                print(f"Completed seed={seed}, model={family}, weights={regime}", flush=True)
    metric_df, pred_df = pd.DataFrame(metrics), pd.DataFrame(predictions)
    metric_df.to_csv(out / "metrics_by_seed.csv", index=False)
    pred_df.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(attributions).to_csv(out / "grouped_shap_values.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "additivity_checks.csv", index=False)
    pd.DataFrame(dossiers).to_csv(out / "decision_dossiers.csv", index=False)
    pd.DataFrame(fit_samples).to_csv(out / "training_bootstrap_indices.csv", index=False)
    table1 = metric_df.groupby(["model", "weighting", "policy"], sort=True).agg(
        auroc=("auroc", "mean"), average_precision=("average_precision", "mean"),
        brier=("brier", "mean"), approval_pct=("approval_pct", "mean"),
        scenario_loss=("scenario_loss", "mean"), loss_sd=("scenario_loss", "std"),
        model_expected_loss=("model_expected_loss", "mean")).reset_index()
    table1.to_csv(out / "table1_performance.csv", index=False)
    avg = pred_df.groupby(["model", "weighting", "policy", "row_id"], sort=True).scenario_loss.mean()
    baseline = avg.loc[("LR", "standard", "U")]
    ci_rows = []
    for family in cfg["model_families"]:
        for regime in cfg["weighting_regimes"]:
            for policy in ("U", "E"):
                delta = avg.loc[(family, regime, policy)].reindex(baseline.index) - baseline
                lo, hi = paired_ci(delta, cfg["bootstrap_reps"], cfg["bootstrap_seed"])
                ci_rows.append({"model": family, "weighting": regime, "policy": policy,
                                "delta_loss": float(delta.mean()), "ci_low": float(lo), "ci_high": float(hi)})
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out / "bootstrap_all_policy_comparisons.csv", index=False)
    ci_df.query("policy == 'E'").to_csv(out / "table2_loss_comparisons.csv", index=False)
    pairs, per_record, table3 = stability_rows(fits, te, margin[te] / (margin[te] + loss[te]))
    pairs.to_csv(out / "stability_pairs.csv", index=False)
    per_record.to_csv(out / "stability_by_record.csv", index=False)
    table3.to_csv(out / "table3_explanation_stability.csv", index=False)
    rf = fits[("RF", "standard")]
    ensemble_p = np.mean([entry["p_test"] for entry in rf], axis=0)
    tau = margin[te] / (margin[te] + loss[te])
    instability = per_record.query("model == 'RF' and weighting == 'standard'").set_index("row_id").loc[te, "instability"].to_numpy()
    review, routing, review_cases = review_analysis(te, ensemble_p, tau, instability, y[te], margin[te], loss[te], cfg)
    review.to_csv(out / "review_all_scenarios.csv", index=False)
    review.query("scenario == 'imperfect'").to_csv(out / "table4_imperfect_review.csv", index=False)
    routing.to_csv(out / "review_routing.csv", index=False)
    review_cases.to_csv(out / "review_case_accounting.csv", index=False)
    sensitivity_analysis(x, fi, te, y, ensemble_p, cfg).to_csv(out / "table5_cost_sensitivity.csv", index=False)
    age_audit(age[te], y[te], ensemble_p, tau, cfg).to_csv(out / "table6_age_audit.csv", index=False)
    pd.DataFrame({"row_id": te, "truth": y[te], "age_audit_only": age[te], "prob_bad": ensemble_p,
                  "economic_threshold": tau, "approve": ensemble_p < tau,
                  "scenario_loss": scenario_loss(ensemble_p < tau, y[te], margin[te], loss[te])}).to_csv(
                      out / "rf_standard_ensemble_predictions.csv", index=False)
    print("Computing 60 LIME explanations (20 records x 3 seeds)...", flush=True)
    lime_records, lime_weights, lime_pairs, lime_summary, lime_ids = lime_diagnostics(x, split, rf[0], prep, cfg)
    lime_records.to_csv(out / "lime_explanations.csv", index=False)
    lime_weights.to_csv(out / "lime_coefficients.csv", index=False)
    lime_pairs.to_csv(out / "lime_seed_overlap.csv", index=False)
    write_json(out / "lime_summary.json", lime_summary)
    write_json(out / "lime_record_ids.json", {"row_ids": [int(i) for i in lime_ids]})
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scipy", "scikit-learn", "shap", "lime", "matplotlib", "joblib"]}
    manifest = {"status": cfg["status"], "python": sys.version, "platform": platform.platform(),
                "package_versions": versions, "data_path": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else data_path.name,
                "data_sha256": digest(data_path), "protocol_sha256": digest(config_path),
                "split_sha256": digest(split_path), "rows": len(y), "bad_credits": int(y.sum()),
                "features": FEATURES, "fitted_models": len(checks), "metric_rows": len(metrics),
                "prediction_rows": len(predictions), "lime_explanations": len(lime_records),
                "preprocessing_fit_rows": len(fi), "max_raw_additivity_error": max(r["raw_additivity_max_error"] for r in checks),
                "max_calibrated_additivity_error": max(r["calibrated_grouped_additivity_max_error"] for r in checks),
                "all_partitions_disjoint": True, "original_numerical_replication_certified": False,
                "training_loss_sd_ddof": 1, "review_instability_undefined_records": int(routing.instability_undefined.sum()),
                "manuscript_sha256": digest(ROOT / "manuscript" / "manuscript_supplied.pdf")}
    write_json(out / "manifest.json", manifest)
    create_reports(out, ROOT)
    print(f"All experiments completed: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/SouthGermanCredit.asc")
    parser.add_argument("--config", type=Path, default=ROOT / "config/protocol.json")
    parser.add_argument("--splits", type=Path, default=ROOT / "config/frozen_split.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/new_run")
    args = parser.parse_args()
    run(args.data.resolve(), args.config.resolve(), args.splits.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
