import numpy as np

from src.diagnostics import _cost_tables


def test_cost_scenarios_use_only_development_and_move_threshold_with_costs():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([.05, .20, .45, .40, .70, .95])
    curves, choices = _cost_tables(y, p)
    assert set(choices.data_scope) == {"development_oof_only"}
    assert len(curves) == 3 * 91
    threshold = choices.set_index("scenario").threshold
    assert threshold["fn_5_fp_1"] <= threshold["fn_2_fp_3"]
