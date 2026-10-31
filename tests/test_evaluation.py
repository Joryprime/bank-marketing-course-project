import numpy as np
import pandas as pd
import pytest

from src.evaluation import capacity_table, score, top_mask


def test_top_mask_has_exact_ceiling_capacity_and_is_reproducible():
    p = np.ones(21)
    a = top_mask(p, fraction=.1, seed=7)
    b = top_mask(p, fraction=.1, seed=7)
    assert a.sum() == 3
    assert np.array_equal(a, b)


def test_perfect_ranking_metrics():
    y = np.array([0] * 90 + [1] * 10)
    p = np.arange(100) / 100
    m = score(y, p)
    assert m["ap"] == pytest.approx(1.0)
    assert m["top10_hits"] == 10
    assert m["top10_lift"] == 10.0


def test_capacity_table_states_denominators():
    y = np.array([0, 0, 1, 1] * 25)
    frame = capacity_table(y, {"model": y.astype(float)}, "test", 3)
    row = frame[(frame.strategy == "model") & (frame.capacity == .1)].iloc[0]
    assert row.contacts == 10
    assert row.total_positives == 50
    assert row.hits == 10
