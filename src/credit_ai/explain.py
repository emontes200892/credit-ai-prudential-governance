"""Grouped exact score attributions and LIME diagnostics."""
import itertools

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from .core import (FEATURES, NUMERIC, finite_mean, jaccard, positive_reasons,
                   raw_score, transformed_groups)


def grouped_shap(model, family, prep, calibration, z_fit, z_test):
    """SHAP for calibrated logit a*raw_score+b; keep policy terms separate."""
    if family == "LR":
        # Exact linear interventional SHAP, using all 550 fitting-set means.
        background_mean = z_fit.mean(axis=0)
        phi = (z_test - background_mean) * model.coef_[0]
        base = float(model.intercept_[0] + background_mean @ model.coef_[0])
    else:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent",
                                      model_output="raw")
        phi = explainer.shap_values(z_test, check_additivity=True)
        base = explainer.expected_value
        if family == "RF":
            # sklearn RF TreeSHAP's raw output is the two-class probability vector.
            if isinstance(phi, list):
                phi = phi[1]
            else:
                phi = np.asarray(phi)[:, :, 1]
            base = float(np.asarray(base)[1])
        else:
            phi = np.asarray(phi)
            base = float(np.asarray(base).reshape(-1)[0])
    raw = raw_score(model, z_test)
    raw_error = float(np.max(np.abs(base + phi.sum(axis=1) - raw)))
    if raw_error > 1e-6:
        raise AssertionError(f"SHAP raw additivity failed: {raw_error}")
    slope = float(calibration.coef_[0, 0])
    intercept = float(calibration.intercept_[0])
    calibrated_phi = phi * slope
    calibrated_base = slope * base + intercept
    group_labels = transformed_groups(prep)
    grouped = np.column_stack([
        calibrated_phi[:, [j for j, group in enumerate(group_labels) if group == name]].sum(axis=1)
        for name in FEATURES
    ])
    calibrated_logit = calibration.decision_function(raw.reshape(-1, 1))
    calibrated_error = float(np.max(np.abs(calibrated_base + grouped.sum(axis=1) - calibrated_logit)))
    if calibrated_error > 1e-6:
        raise AssertionError("Grouped/calibrated SHAP additivity failed")
    return grouped, calibrated_base, raw_error, calibrated_error


def stability_rows(fits, ids, tau):
    pairs, per_record, summary = [], [], []
    for (family, regime), entries in fits.items():
        local_pairs = []
        for left, right in itertools.combinations(entries, 2):
            a = left["p_test"] < tau
            b = right["p_test"] < tau
            for i, row_id in enumerate(ids):
                local_pairs.append({
                    "model": family, "weighting": regime, "row_id": int(row_id),
                    "seed_left": left["seed"], "seed_right": right["seed"],
                    "same_action": bool(a[i] == b[i]),
                    "joint_denial": bool(not a[i] and not b[i]),
                    "flip": bool(a[i] != b[i]),
                    "jaccard": jaccard(left["reasons"][i], right["reasons"][i]),
                })
        df = pd.DataFrame(local_pairs)
        pairs.extend(local_pairs)
        summary.append({
            "model": family, "weighting": regime,
            "same_action_overlap": finite_mean(df.loc[df.same_action, "jaccard"]),
            "joint_denial_overlap": finite_mean(df.loc[df.joint_denial, "jaccard"]),
            "flip_rate": float(df.flip.mean()), "all_pair_count": len(df),
            "same_action_count": int(df.same_action.sum()),
            "same_action_valid_overlap_count": int(df.loc[df.same_action, "jaccard"].notna().sum()),
            "joint_denial_count": int(df.joint_denial.sum()),
            "joint_denial_valid_overlap_count": int(df.loc[df.joint_denial, "jaccard"].notna().sum()),
            "undefined_overlap_count": int(df.jaccard.isna().sum()),
        })
        for row_id, g in df.groupby("row_id", sort=True):
            mean_overlap = finite_mean(g.jaccard)
            per_record.append({"model": family, "weighting": regime, "row_id": int(row_id),
                               "mean_pairwise_overlap": mean_overlap,
                               "instability": 1 - mean_overlap,
                               "valid_pairs": int(g.jaccard.notna().sum()),
                               "flip_rate": float(g.flip.mean())})
    return pd.DataFrame(pairs), pd.DataFrame(per_record), pd.DataFrame(summary)


def lime_diagnostics(x, split, entry, prep, cfg):
    ids = split["test"]
    rng = np.random.default_rng(cfg["lime_record_seed"])
    positions = np.sort(rng.choice(len(ids), size=cfg["lime_record_count"], replace=False))
    categoricals = [i for i, f in enumerate(FEATURES) if f not in NUMERIC]

    def predict(values):
        # Numeric dimensions remain continuous; category dimensions are sampled as codes.
        frame = pd.DataFrame(values, columns=FEATURES)
        raw = raw_score(entry["model"], prep.transform(frame))
        return entry["calibration"].predict_proba(raw.reshape(-1, 1))

    rows, coefficients = [], []
    for seed in cfg["lime_seeds"]:
        # One explainer per seed, processing the same ordered record subset.
        explainer = LimeTabularExplainer(
            x.iloc[split["fit"]].to_numpy(float), feature_names=FEATURES,
            categorical_features=categoricals, class_names=["good", "bad"],
            mode="classification", discretize_continuous=False,
            feature_selection="none", sample_around_instance=False, random_state=seed)
        for pos in positions:
            exp = explainer.explain_instance(x.iloc[ids[pos]].to_numpy(float), predict,
                                            labels=(1,), num_features=len(FEATURES),
                                            num_samples=cfg["lime_samples"])
            weights = np.zeros(len(FEATURES))
            for feature_id, weight in exp.local_exp[1]:
                weights[feature_id] = weight
            reasons = positive_reasons(weights)
            rows.append({"row_id": int(ids[pos]), "explainer_seed": seed,
                         "weighted_neighborhood_r2": float(exp.score),
                         "positive_top3": "|".join(reasons),
                         "shap_positive_top3": "|".join(entry["reasons"][pos]),
                         "cross_method_overlap": jaccard(reasons, entry["reasons"][pos]),
                         "local_prediction": float(np.asarray(exp.local_pred).reshape(-1)[0]),
                         "calibrated_prob_bad": float(entry["p_test"][pos]),
                         "lime_intercept": float(exp.intercept[1]),
                         "num_samples": cfg["lime_samples"]})
            coefficients.extend({"row_id": int(ids[pos]), "explainer_seed": seed,
                                 "feature": f, "coefficient": float(w)}
                                for f, w in zip(FEATURES, weights))
    df = pd.DataFrame(rows)
    overlap = []
    for row_id, g in df.groupby("row_id", sort=True):
        for a, b in itertools.combinations(g.itertuples(), 2):
            aset = a.positive_top3.split("|") if a.positive_top3 else []
            bset = b.positive_top3.split("|") if b.positive_top3 else []
            overlap.append({"row_id": int(row_id), "seed_left": a.explainer_seed,
                            "seed_right": b.explainer_seed, "jaccard": jaccard(aset, bset)})
    paired = pd.DataFrame(overlap)
    summary = {"weighted_neighborhood_r2": float(df.weighted_neighborhood_r2.mean()),
               "explainer_seed_overlap": finite_mean(paired.jaccard),
               "cross_method_overlap": finite_mean(df.cross_method_overlap),
               "explanation_count": len(df), "record_count": len(positions),
               "valid_seed_overlap_pairs": int(paired.jaccard.notna().sum()),
               "valid_cross_method_overlap_count": int(df.cross_method_overlap.notna().sum()),
               "output_object": "LIME: calibrated bad-credit probability; SHAP: calibrated logit"}
    return df, pd.DataFrame(coefficients), paired, summary, ids[positions].tolist()
