"""Generate manuscript-linked tables, figures and honest numerical comparisons."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_reports(out, root):
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    t1 = pd.read_csv(out / "table1_performance.csv")
    t3 = pd.read_csv(out / "table3_explanation_stability.csv")
    points = t1.query("policy == 'E'").merge(t3, on=["model", "weighting"])
    fig, ax = plt.subplots(figsize=(8.3, 5.2), layout="constrained")
    for row in points.itertuples():
        ax.scatter(row.scenario_loss, row.joint_denial_overlap, s=45)
        ax.annotate(f"{row.model}-{row.weighting}", (row.scenario_loss, row.joint_denial_overlap),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean realized-label scenario loss (lower is better)",
           ylabel="Joint-denial positive top-3 SHAP overlap",
           title="Reconstructed run: cost and explanation stability")
    ax.margins(x=.18, y=.12)
    ax.grid(alpha=.18)
    fig.savefig(figures / "figure1_cost_stability.png", dpi=180)
    fig.savefig(figures / "figure1_cost_stability.pdf")
    plt.close(fig)
    review = pd.read_csv(out / "table4_imperfect_review.csv")
    fig, ax = plt.subplots(figsize=(7.3, 4.6), layout="constrained")
    for name, group in review.groupby("routing", sort=True):
        group = group.sort_values("budget")
        ax.plot(group.budget * 100, group.delta_loss, marker="o", label=name)
    ax.axhline(0, c="gray", lw=.8)
    ax.set(xlabel="Review capacity (%)", ylabel="Change in mean scenario loss",
           title="Reconstructed run: hypothetical imperfect review", xticks=[5, 10, 20])
    ax.legend(frameon=False)
    ax.grid(alpha=.18)
    fig.savefig(figures / "figure2_imperfect_review.png", dpi=180)
    fig.savefig(figures / "figure2_imperfect_review.pdf")
    plt.close(fig)
    for p in sorted(out.glob("table*.csv")):
        df = pd.read_csv(p)
        # Plain tabular fragments; no optional pandas/Jinja dependency needed.
        def tex(value):
            if isinstance(value, float):
                return f"{value:.6f}" if np.isfinite(value) else "--"
            return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        lines = [r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\hline",
                 " & ".join(tex(c) for c in df.columns) + r" \\", r"\hline"]
        lines.extend(" & ".join(tex(v) for v in row) + r" \\" for row in df.itertuples(index=False, name=None))
        lines += [r"\hline", r"\end{tabular}"]
        (out / (p.stem + ".tex")).write_text("\n".join(lines) + "\n")
    comparisons = []

    def add(table, keys, reported_file, actual_file, metrics):
        expected = pd.read_csv(root / "manuscript" / reported_file)
        actual = pd.read_csv(out / actual_file)
        merged = expected.merge(actual, on=keys, suffixes=("_reported", "_run"), validate="one_to_one")
        if len(merged) != len(expected):
            raise AssertionError(f"Missing manuscript comparison rows in {table}")
        for _, row in merged.iterrows():
            for metric, precision in metrics.items():
                reported, generated = float(row[metric + "_reported"]), float(row[metric + "_run"])
                tolerance = .5 * 10 ** (-precision) + 1e-12
                comparisons.append({"table": table, "configuration": " / ".join(str(row[k]) for k in keys),
                                    "metric": metric, "reported": reported, "reconstructed": generated,
                                    "difference": generated - reported, "rounding_tolerance": tolerance,
                                    "matches_reported_rounding": abs(generated - reported) <= tolerance})

    add("Table 1", ["model", "weighting", "policy"], "reported_table1.csv", "table1_performance.csv",
        {"auroc": 3, "average_precision": 3, "brier": 3, "approval_pct": 1, "scenario_loss": 4, "loss_sd": 4})
    add("Table 2", ["model", "weighting"], "reported_table2.csv", "table2_loss_comparisons.csv",
        {"delta_loss": 4, "ci_low": 4, "ci_high": 4})
    add("Table 3", ["model", "weighting"], "reported_table3.csv", "table3_explanation_stability.csv",
        {"same_action_overlap": 3, "joint_denial_overlap": 3, "flip_rate": 3})
    add("Table 4", ["routing", "budget"], "reported_table4.csv", "table4_imperfect_review.csv", {"delta_loss": 4})
    add("Table 5", ["loss_coefficient", "margin_multiplier"], "reported_table5.csv", "table5_cost_sensitivity.csv",
        {"approval_pct": 1, "scenario_loss": 4})
    add("Table 6", ["age_group"], "reported_table6.csv", "table6_age_audit.csv",
        {"n": 0, "approval_pct": 1, "bad_approval_pct": 1, "good_rejection_pct": 1})
    lime = json.loads((out / "lime_summary.json").read_text())
    for metric, reported in {"weighted_neighborhood_r2": .491, "explainer_seed_overlap": .783,
                             "cross_method_overlap": .522}.items():
        generated = lime[metric]
        comparisons.append({"table": "Section 6.2 LIME", "configuration": "RF standard seed 11", "metric": metric,
                            "reported": reported, "reconstructed": generated, "difference": generated - reported,
                            "rounding_tolerance": .0005,
                            "matches_reported_rounding": abs(generated - reported) <= .0005})
    cmp = pd.DataFrame(comparisons)
    cmp.to_csv(out / "manuscript_comparison.csv", index=False)
    manifest = json.loads((out / "manifest.json").read_text())
    count = int(cmp.matches_reported_rounding.sum())
    (out / "RUN_REPORT.md").write_text(
        "# Verified reconstructed run\n\n"
        "This is a complete execution of the reconstructed historical-credit protocol. "
        "It is not recovery of the original run underlying the supplied manuscript.\n\n"
        f"- Fitted models: {manifest['fitted_models']}; test prediction/action rows: {manifest['prediction_rows']}.\n"
        f"- LIME explanations: {manifest['lime_explanations']}.\n"
        f"- Maximum raw-score SHAP additivity error: {manifest['max_raw_additivity_error']:.3g}.\n"
        f"- Maximum calibrated/grouped additivity error: {manifest['max_calibrated_additivity_error']:.3g}.\n"
        f"- Entries agreeing within reported rounding: {count}/{len(cmp)}. "
        "Individual rounded agreements are not evidence that the original split was recovered.\n\n"
        "Inspect manuscript_comparison.csv before reusing any claim. Tables and figures in this "
        "directory are generated from current predictions and attributions, not from paper summaries. "
        "The manuscript PDF was preserved unchanged. HMDA, LendingClub and legal-dossier validation "
        "are planned work and have no empirical outputs here.\n")
