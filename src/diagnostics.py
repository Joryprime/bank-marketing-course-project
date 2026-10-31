"""Development-only diagnostics that never read the independent or hidden sets."""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.modeling import CATEGORICAL, FEATURES, NUMERIC, build_hist_model, build_model


DIAGNOSTIC_FILES = [
    "ablation_features.csv",
    "calibration_bins.csv",
    "calibration_summary.json",
    "feature_importance.csv",
    "threshold_cost_curves.csv",
    "threshold_cost_scenarios.csv",
]


def _evaluate_feature_set(dev, folds, numeric, categorical, seed):
    features = list(numeric) + list(categorical)
    ap, auc = [], []
    for tr, va in folds:
        model = build_hist_model(seed, numeric, categorical)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.fit(dev.iloc[tr][features], dev.y.iloc[tr])
        probability = model.predict_proba(dev.iloc[va][features])[:, 1]
        ap.append(average_precision_score(dev.y.iloc[va], probability))
        auc.append(roc_auc_score(dev.y.iloc[va], probability))
    return {
        "ap_mean": float(np.mean(ap)), "ap_std": float(np.std(ap, ddof=1)),
        "auc_mean": float(np.mean(auc)), "auc_std": float(np.std(auc, ddof=1)),
    }


def _cost_tables(y, probability):
    scenarios = [
        ("fn_5_fp_1", 5.0, 1.0),
        ("fn_2_fp_3", 2.0, 3.0),
        ("fn_1_fp_1", 1.0, 1.0),
    ]
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    curves, choices = [], []
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    for name, fn_cost, fp_cost in scenarios:
        scenario_rows = []
        for threshold in thresholds:
            predicted = probability >= threshold
            fp = int((predicted & (y == 0)).sum())
            fn = int((~predicted & (y == 1)).sum())
            tp = int((predicted & (y == 1)).sum())
            tn = int((~predicted & (y == 0)).sum())
            row = dict(scenario=name, fn_cost=fn_cost, fp_cost=fp_cost,
                       threshold=float(threshold), development_cost=float(fn_cost * fn + fp_cost * fp),
                       development_tn=tn, development_fp=fp, development_fn=fn, development_tp=tp)
            curves.append(row)
            scenario_rows.append(row)
        best = min(scenario_rows, key=lambda row: (row["development_cost"], row["threshold"]))
        choices.append({**best, "data_scope": "development_oof_only",
                        "assumption": "Illustrative unit weights, not observed monetary costs"})
    return pd.DataFrame(curves), pd.DataFrame(choices)


def run_development_diagnostics(dev, folds, oof_probability, selected_model, out, seed):
    """Generate explanatory evidence before the model lock and test evaluation."""
    out = Path(out)
    baseline_ap, baseline_auc = [], []
    for _, va in folds:
        baseline_ap.append(average_precision_score(dev.y.iloc[va], oof_probability[va]))
        baseline_auc.append(roc_auc_score(dev.y.iloc[va], oof_probability[va]))
    rows = [{
        "scenario": "baseline_11_features", "ap_mean": np.mean(baseline_ap),
        "ap_std": np.std(baseline_ap, ddof=1), "auc_mean": np.mean(baseline_auc),
        "auc_std": np.std(baseline_auc, ddof=1), "n_features": len(FEATURES),
        "data_scope": "development_grouped_cv", "used_for_selection": False,
        "interpretation": "Locked contact-time feature allowlist",
    }]
    plans = [
        ("add_duration", NUMERIC + ["duration"], CATEGORICAL,
         "Timing-violation diagnostic; duration remains excluded"),
        ("drop_balance", [c for c in NUMERIC if c != "balance"], CATEGORICAL,
         "Retrained feature-removal sensitivity"),
        ("drop_poutcome", NUMERIC, [c for c in CATEGORICAL if c != "poutcome"],
         "Retrained feature-removal sensitivity"),
    ]
    for scenario, numeric, categorical, interpretation in plans:
        result = _evaluate_feature_set(dev, folds, numeric, categorical, seed)
        rows.append({"scenario": scenario, **result,
                     "n_features": len(numeric) + len(categorical),
                     "data_scope": "development_grouped_cv", "used_for_selection": False,
                     "interpretation": interpretation})
    ablation = pd.DataFrame(rows)
    ablation.to_csv(out / "ablation_features.csv", index=False)

    tr, va = folds[0]
    model = build_model(selected_model, seed)
    model.fit(dev.iloc[tr][FEATURES], dev.y.iloc[tr])
    validation = dev.iloc[va]
    if len(validation) > 40000:
        chosen = np.random.default_rng(seed).choice(len(validation), 40000, replace=False)
        validation = validation.iloc[chosen]
    importance = permutation_importance(
        model, validation[FEATURES], validation.y, scoring="average_precision",
        n_repeats=5, random_state=seed, n_jobs=1,
    )
    pd.DataFrame({
        "feature": FEATURES,
        "importance_mean_ap_decrease": importance.importances_mean,
        "importance_std": importance.importances_std,
        "fold": 1, "validation_sample_n": len(validation),
        "data_scope": "development_fold_1_validation",
    }).sort_values("importance_mean_ap_decrease", ascending=False).to_csv(
        out / "feature_importance.csv", index=False)

    probability = np.asarray(oof_probability, dtype=float)
    bins = pd.qcut(pd.Series(probability), 10, duplicates="drop")
    calibration = (pd.DataFrame({"probability": probability, "y": dev.y.to_numpy(), "bin": bins})
                   .groupby("bin", observed=True)
                   .agg(n=("y", "size"), mean_prediction=("probability", "mean"),
                        observed_rate=("y", "mean")).reset_index(drop=True))
    calibration["gap"] = calibration.mean_prediction - calibration.observed_rate
    calibration["absolute_gap"] = calibration.gap.abs()
    calibration.to_csv(out / "calibration_bins.csv", index=False)
    summary = {
        "model": selected_model, "data_scope": "development_oof_only",
        "brier": float(brier_score_loss(dev.y, probability)),
        "weighted_mean_absolute_bin_gap": float(np.average(calibration.absolute_gap, weights=calibration.n)),
        "maximum_absolute_bin_gap": float(calibration.absolute_gap.max()),
        "caution": "Coarse decile diagnostic; not proof of production calibration",
    }
    import json
    (out / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    curves, scenarios = _cost_tables(dev.y, probability)
    curves.to_csv(out / "threshold_cost_curves.csv", index=False)
    scenarios.to_csv(out / "threshold_cost_scenarios.csv", index=False)

    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(ablation.scenario, ablation.ap_mean, yerr=ablation.ap_std, capsize=4,
           color=["#0f766e", "#b91c1c", "#64748b", "#64748b"])
    ax.set(ylabel="Grouped 5-fold AP", title="Development-only feature sensitivity")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figdir / "ablation_features.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, group in curves.groupby("scenario"):
        relative = group.development_cost / group.development_cost.max()
        ax.plot(group.threshold, relative, linewidth=2, label=name)
    ax.set(xlabel="Decision threshold", ylabel="Relative development cost",
           title="Illustrative costs change the preferred threshold")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figdir / "cost_curves.png", dpi=160); plt.close(fig)

    return DIAGNOSTIC_FILES
