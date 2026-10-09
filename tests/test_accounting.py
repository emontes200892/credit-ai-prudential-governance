import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_ai.core import (EXCLUDED, FEATURES, financial_terms, jaccard, load_data,
                           paired_ci, percentile_closeness, review_loss, route_indices,
                           scenario_loss, select_threshold, training_weights, validate_split)
from credit_ai.analysis import review_analysis

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/protocol.json").read_text())


def test_uci_target_recode_and_feature_exclusions():
    x, y, age, raw = load_data(ROOT / "data/raw/SouthGermanCredit.asc")
    assert np.array_equal(y, 1 - raw.credit_risk.to_numpy())
    assert y.sum() == 300 and len(y) == 1000
    assert len(x.columns) == 17 and set(EXCLUDED).isdisjoint(x.columns)
    assert np.array_equal(age, raw.age)
    csv_x, csv_y, csv_age, _ = load_data(ROOT / "data/processed/south_german_bad_credit.csv")
    pd.testing.assert_frame_equal(x, csv_x)
    np.testing.assert_array_equal(y, csv_y)


def test_frozen_partitions_and_overlap_rejection():
    _, y, _, _ = load_data(ROOT / "data/raw/SouthGermanCredit.asc")
    split = {k: np.asarray(v) for k, v in json.loads((ROOT / "config/frozen_split.json").read_text())["indices"].items()}
    validate_split(split, y)
    broken = {k: v.copy() for k, v in split.items()}
    same_label_fit = next(i for i in split['fit'] if y[i] == y[split['test'][0]])
    broken['test'][0] = same_label_fit
    with pytest.raises(ValueError):
        validate_split(broken, y)


def test_loss_matrix_and_threshold_tie():
    np.testing.assert_array_equal(scenario_loss([True, True, False, False], np.array([0, 1, 0, 1]),
                                              np.array([2, 2, 2, 2]), np.array([6, 6, 6, 6])), [-2, 6, 0, 0])
    # All thresholds reject p=1; first minimum must be the smallest threshold.
    assert select_threshold(np.array([1.]), np.array([1]), np.array([2.]), np.array([6.]), .005) == 0.


def test_economic_policy_is_expected_loss_minimizer():
    rng = np.random.default_rng(17)
    p = rng.uniform(.001, .999, 1000)
    margin, loss = rng.uniform(.01, 3, (2, 1000))
    tau = margin / (margin + loss)
    g = p * loss - (1 - p) * margin
    d = np.log(p / (1 - p)) + np.log(loss) - np.log(margin)
    np.testing.assert_array_equal(p < tau, g < 0)
    np.testing.assert_array_equal(p < tau, d < 0)
    np.testing.assert_array_equal(np.array([.25]) < 2 / (2 + 6), [False])


def test_class_and_idcs_weights():
    y = np.array([0, 0, 0, 1])
    margin, loss = np.array([1., 2, 3, 4]), np.array([8., 9, 10, 11])
    balanced = training_weights(y, margin, loss, 'balanced')
    assert balanced[y == 0].sum() == balanced[y == 1].sum()
    idcs = training_weights(y, margin, loss, 'idcs')
    np.testing.assert_allclose(idcs, np.array([1., 2, 3, 11]) / 4.25)
    assert idcs.mean() == 1.


@pytest.mark.parametrize('c,r', [(0., 0.), (.5, .05), (1., 0.)])
def test_review_benefit_identity_and_oracle(c, r):
    y = np.array([0, 1, 0, 1])
    approve = np.array([True, False, False, True])
    margin, loss = np.full(4, 2.), np.full(4, 6.)
    auto = scenario_loss(approve, y, margin, loss)
    human = review_loss(approve, y, margin, loss, c, r, .01)
    wrong = approve != (y == 0)
    expected_benefit = (c * wrong - r * (~wrong)) * np.where(y == 1, loss, margin) - .01
    np.testing.assert_allclose(auto - human, expected_benefit)
    if c == r == 0:
        np.testing.assert_allclose(human, auto + .01)
    if c == 1 and r == 0:
        np.testing.assert_allclose(human, -margin * (1 - y) + .01)


def test_random_review_expectation_and_label_free_selection():
    n = 200
    ids = np.arange(n)
    p = np.linspace(.01, .8, n)
    tau, inst = np.full(n, .25), np.linspace(0, 1, n)
    y, margin, loss = ids % 2, np.ones(n), np.full(n, 3.)
    review, routing, cases = review_analysis(ids, p, tau, inst, y, margin, loss, CFG)
    _, other_routing, _ = review_analysis(ids, p, tau, inst, 1-y, margin, loss, CFG)
    pd.testing.assert_frame_equal(routing, other_routing)
    g = cases.query("scenario == 'imperfect'").review_delta.mean()
    random = review.query("scenario == 'imperfect' and routing == 'random'")
    np.testing.assert_allclose(random.delta_loss, random.budget * g)


def test_reason_overlap_missing_and_routing_ties():
    assert np.isnan(jaccard([], []))
    assert np.isnan(jaccard([], ['a']))
    assert jaccard(['a', 'b'], ['b', 'c']) == 1/3
    assert jaccard(['a'], ['a']) == 1
    np.testing.assert_array_equal(route_indices([1, 1, .1], [6, 2, 1], 1), [1])
    score = percentile_closeness(np.array([.25, .75, .99]), np.array([.5, .5, .5]))
    assert score[0] == score[1] and score[0] > score[2]


def test_bootstrap_pairing_constant_shift():
    lo, hi = paired_ci(np.full(200, -.12), 2000, 2026)
    assert lo == pytest.approx(-.12) and hi == pytest.approx(-.12)
