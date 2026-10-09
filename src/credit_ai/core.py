"""Data validation, policy accounting and inference primitives.

Row IDs are zero-based physical data-row positions (header excluded).
Only bad_credit is coded 1=bad. UCI's kredit/credit_risk is 0=bad.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GERMAN = ["laufkont", "laufzeit", "moral", "verw", "hoehe", "sparkont",
          "beszeit", "rate", "famges", "buerge", "wohnzeit", "verm", "alter",
          "weitkred", "wohn", "bishkred", "beruf", "pers", "telef", "gastarb", "kredit"]
ENGLISH = ["status", "duration", "credit_history", "purpose", "amount", "savings",
           "employment_duration", "installment_rate", "personal_status_sex",
           "other_debtors", "present_residence", "property", "age",
           "other_installment_plans", "housing", "number_credits", "job",
           "people_liable", "telephone", "foreign_worker", "credit_risk"]
EXCLUDED = ["age", "personal_status_sex", "foreign_worker"]
NUMERIC = ["amount", "duration"]
FEATURES = [v for v in ENGLISH[:-1] if v not in EXCLUDED]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+" if path.suffix.lower() == ".asc" else ",")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=dict(zip(GERMAN, ENGLISH)))
    target = "bad_credit" if "bad_credit" in df else "credit_risk"
    required = set(ENGLISH[:-1]) | {target}
    if set(df) != required:
        raise ValueError(f"Schema mismatch; missing={required-set(df)}, extra={set(df)-required}")
    df = df.apply(pd.to_numeric, errors="raise")
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Missing/nonfinite values are not permitted in this frozen benchmark")
    if not df.eq(np.floor(df)).all().all():
        raise ValueError("Source benchmark fields must be integer-valued")
    df = df.astype(int)
    if set(df[target].unique()) != {0, 1}:
        raise ValueError("Target must be binary")
    y = df[target].to_numpy() if target == "bad_credit" else 1 - df[target].to_numpy()
    if len(df) != 1000 or y.sum() != 300:
        raise ValueError("Expected 1,000 records, 300 bad and 700 good after documented recoding")
    if (df[NUMERIC] <= 0).any().any():
        raise ValueError("Amount and duration must be positive")
    return df[FEATURES].copy(), y, df["age"].to_numpy(), df


def make_split(y, seeds):
    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=200, stratify=y, random_state=seeds[0])
    fit, rem = train_test_split(train, train_size=550, stratify=y[train], random_state=seeds[1])
    cal, policy = train_test_split(rem, train_size=150, stratify=y[rem], random_state=seeds[2])
    return {"fit": fit, "calibration": cal, "policy": policy, "test": test}


def validate_split(split, y):
    sizes = {"fit": 550, "calibration": 150, "policy": 100, "test": 200}
    if set(split) != set(sizes):
        raise ValueError("Incorrect partition names")
    for k, n in sizes.items():
        a = np.asarray(split[k])
        if len(a) != n or a.dtype.kind not in "iu" or len(set(a)) != n:
            raise ValueError(f"Invalid partition {k}")
        if (a < 0).any() or (a >= len(y)).any() or int(y[a].sum()) != int(n * 0.3):
            raise ValueError(f"Invalid indices/class balance in {k}")
    if len(set(np.concatenate(list(split.values())))) != len(y):
        raise ValueError("Partitions overlap or fail to cover the data")


def financial_terms(x, fit, cfg, loss_coefficient=None, margin_multiplier=1.):
    median = float(np.median(x.iloc[fit]["amount"]))
    exposure = x["amount"].to_numpy(float) / median
    margin = exposure * (cfg["margin_intercept"] + cfg["margin_duration_slope"] *
                         np.minimum(x["duration"].to_numpy(float), cfg["duration_cap"]))
    coefficient = cfg["loss_coefficient"] if loss_coefficient is None else loss_coefficient
    return exposure, margin * margin_multiplier, coefficient * exposure + cfg["loss_fixed_term"]


def make_preprocessor():
    cats = [v for v in FEATURES if v not in NUMERIC]
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ], sparse_threshold=0)


def transformed_groups(prep):
    cats = [v for v in FEATURES if v not in NUMERIC]
    return NUMERIC + [c for c, levels in zip(cats, prep.named_transformers_["categorical"].categories_)
                      for _ in levels]


def training_weights(y, margin, loss, regime):
    if regime == "standard":
        return np.ones(len(y))
    if regime == "balanced":
        counts = np.bincount(y, minlength=2)
        if (counts == 0).any():
            raise ValueError("Training bootstrap contains only one class")
        return len(y) / (2 * counts[y])
    if regime != "idcs":
        raise ValueError(regime)
    w = np.where(y == 1, loss, margin)
    return w / w.mean()


def raw_score(model, x):
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def scenario_loss(approve, y, margin, loss):
    return np.where(approve, np.where(y == 1, loss, -margin), 0.)


def model_expected_loss(approve, p, margin, loss):
    return np.where(approve, p * loss - (1 - p) * margin, 0.)


def select_threshold(p, y, margin, loss, step):
    grid = np.linspace(0., 1., int(round(1 / step)) + 1)
    costs = [scenario_loss(p < t, y, margin, loss).mean() for t in grid]
    return float(grid[np.argmin(costs)])  # first minimum = smallest threshold


def positive_reasons(phi, names=FEATURES, k=3):
    order = sorted(range(len(phi)), key=lambda i: (-float(phi[i]), names[i]))
    return [names[i] for i in order if phi[i] > 0][:k]


def jaccard(a, b):
    a, b = set(a), set(b)
    # Any empty reason list has undefined overlap, including empty vs non-empty.
    if not a or not b:
        return np.nan
    return len(a & b) / len(a | b)


def finite_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def paired_ci(delta, reps, seed):
    delta = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(reps, len(delta)))
    return np.quantile(delta[indices].mean(axis=1), [.025, .975])


def review_loss(approve, y, margin, loss, correction, introduction, cost):
    wrong = np.asarray(approve) != (np.asarray(y) == 0)
    v = wrong * (1 - correction) + (~wrong) * introduction
    stakes = np.where(y == 1, loss, margin)
    return -margin * (1 - y) + v * stakes + cost


def percentile_closeness(p, tau):
    # Average ranks for ties, smallest distance gets largest score.
    n = len(p)
    if n < 2:
        return np.zeros(n)
    ranks = pd.Series(-np.abs(p - tau)).rank(method="average").to_numpy()
    return (ranks - 1) / (n - 1)


def route_indices(scores, ids, count):
    return np.lexsort((np.asarray(ids), -np.asarray(scores)))[:count]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
