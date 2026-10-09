"""Label-free routing, retrospective review accounting and subgroup audit."""
import numpy as np
import pandas as pd

from .core import (financial_terms, percentile_closeness, review_loss,
                   route_indices, scenario_loss)


def review_analysis(ids, p, tau, instability, y, margin, loss, cfg):
    approve = p < tau
    baseline_loss = scenario_loss(approve, y, margin, loss)
    raw_instability = np.asarray(instability, dtype=float)
    missing = ~np.isfinite(raw_instability)
    routing_instability = np.where(missing, cfg["undefined_instability_fill"], raw_instability)
    uncertainty = percentile_closeness(p, tau)
    # No labels or realized costs are used to rank records.
    scores = {"uncertainty": uncertainty, "instability": routing_instability,
              "combined": 0.5 * uncertainty + 0.5 * routing_instability}
    routing = pd.DataFrame({"row_id": ids, "prob_bad": p, "economic_threshold": tau,
                            "distance_to_threshold": np.abs(p - tau),
                            "uncertainty": uncertainty, "raw_instability": raw_instability,
                            "instability_undefined": missing, "instability": routing_instability,
                            "combined": scores["combined"]})
    rows, per_case = [], []
    for name, (c, r) in cfg["review_scenarios"].items():
        reviewed_loss = review_loss(approve, y, margin, loss, c, r, cfg["review_cost"])
        delta = reviewed_loss - baseline_loss
        for i, row_id in enumerate(ids):
            per_case.append({"row_id": int(row_id), "scenario": name, "truth": int(y[i]),
                             "approve": bool(approve[i]), "automated_loss": baseline_loss[i],
                             "reviewed_expected_loss": reviewed_loss[i],
                             "review_delta": delta[i], "review_benefit": -delta[i]})
        for budget in cfg["review_budgets"]:
            count = int(round(len(ids) * budget))
            for method in ("random", "uncertainty", "instability", "combined"):
                if method == "random":
                    inclusion = np.full(len(ids), count / len(ids))
                else:
                    inclusion = np.zeros(len(ids))
                    inclusion[route_indices(scores[method], ids, count)] = 1.
                change = float(np.mean(inclusion * delta))
                rows.append({"scenario": name, "correction_fraction": c, "error_introduction": r,
                             "review_cost": cfg["review_cost"], "routing": method, "budget": budget,
                             "review_count": count, "baseline_loss": float(baseline_loss.mean()),
                             "reviewed_loss": float(baseline_loss.mean() + change),
                             "delta_loss": change,
                             "selection_expectation": method == "random"})
                routing[f"inclusion_{method}_{budget:.2f}"] = inclusion
    result = pd.DataFrame(rows)
    check = result.query("scenario == 'no_improvement'")
    if not np.allclose(check.delta_loss, check.budget * cfg["review_cost"], atol=1e-12):
        raise AssertionError("No-improvement review-cost accounting failed")
    return result, routing, pd.DataFrame(per_case)


def sensitivity_analysis(x, fit, ids, y, p, cfg):
    rows = []
    for lc in cfg["sensitivity_loss_coefficients"]:
        for mm in cfg["sensitivity_margin_multipliers"]:
            exposure, margin, loss = financial_terms(x, fit, cfg, lc, mm)
            tau = margin[ids] / (margin[ids] + loss[ids])
            approve = p < tau
            sl = scenario_loss(approve, y[ids], margin[ids], loss[ids])
            rows.append({"loss_coefficient": lc, "margin_multiplier": mm,
                         "approval_pct": float(100 * approve.mean()),
                         "scenario_loss": float(sl.mean()), "models_retrained": False})
    return pd.DataFrame(rows)


def age_audit(age, y, p, tau, cfg):
    approve = p < tau
    rows = []
    for label, mask in [("age <25", age < cfg["age_cutoff"]),
                        ("age >=25", age >= cfg["age_cutoff"])]:
        bad, good = mask & (y == 1), mask & (y == 0)
        rows.append({"age_group": label, "n": int(mask.sum()), "bad_n": int(bad.sum()),
                     "good_n": int(good.sum()), "approval_n": int(approve[mask].sum()),
                     "bad_approval_n": int(approve[bad].sum()),
                     "good_rejection_n": int((~approve[good]).sum()),
                     "approval_pct": float(100 * approve[mask].mean()) if mask.any() else np.nan,
                     "bad_approval_pct": float(100 * approve[bad].mean()) if bad.any() else np.nan,
                     "good_rejection_pct": float(100 * (~approve[good]).mean()) if good.any() else np.nan})
    return pd.DataFrame(rows)
