"""One entry point: verify data, develop, lock, evaluate, refit, export.

No hidden label, external answer, or source ID can enter training.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src.modeling import FEATURES, EXCLUDED, MODELS, build_model
from src.evaluation import score, top_mask, capacity_table, threshold_table, subgroup_table
from src.diagnostics import DIAGNOSTIC_FILES, run_development_diagnostics

SEED = 20261022
TRAIN_HASH = "921e6430ae470959b5795c075736162e320d891bece827d905fe6f3483b9031a"
HIDDEN_HASH = "4169285aa7994d4b79bfb3894cbc0012fe2bc7fbc17a464d12d1b2aa7deacf6f"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def verify(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"Missing or different course release file: {path}")


def code_hashes():
    files = ["main.py", "src/modeling.py", "src/evaluation.py", "src/diagnostics.py", "docs/protocol.md"]
    return {f: sha(ROOT/f) for f in files}


def load_train(data_dir):
    path = data_dir / "student_train.parquet"
    verify(path, TRAIN_HASH)
    d = pd.read_parquet(path)
    if len(d) != 600000 or set(d.columns) != set(FEATURES) | set(EXCLUDED):
        raise ValueError("Course train schema or row count mismatch")
    if set(d.y.unique()) != {0, 1} or d.course_row_id.isna().any() or not d.course_row_id.is_unique:
        raise ValueError("Target or row ID contract failed")
    return d


def groups_for(d):
    # Exact tuples, no 64-bit hash collision and no use of y/row IDs.
    return pd.factorize(pd.MultiIndex.from_frame(d[FEATURES]), sort=False)[0]


def fit_predict(name, train, valid, seed):
    model = build_model(name, seed)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train[FEATURES], train.y)
    notes = sorted(set(str(w.message) for w in caught))
    # Fail visibly rather than silently present an unconverged linear baseline.
    if any("converge" in w.lower() for w in notes):
        raise RuntimeError(f"Training convergence warning: {notes}")
    p = model.predict_proba(valid[FEATURES])[:, 1]
    return model, p, time.perf_counter()-start, notes


def develop(d, out):
    if (out/"model_lock.json").exists():
        raise ValueError("Existing locked run: use --phase finalize or a fresh --output directory")
    started = now()
    log("Generating feature-profile groups and independent holdout")
    groups = groups_for(d)
    dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(d, d.y, groups))
    dev, test = d.iloc[dev_idx].reset_index(drop=True), d.iloc[test_idx].reset_index(drop=True)
    dg, tg = groups[dev_idx], groups[test_idx]
    assert not np.intersect1d(dg, tg).size
    split = pd.DataFrame({"course_row_id": d.course_row_id, "partition": "development", "group": groups})
    split.loc[test_idx, "partition"] = "independent_test"
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    log("Building stratified grouped five-fold splits (this can take a moment)")
    folds = list(cv.split(dev[FEATURES], dev.y, dg))
    oof = {name: np.full(len(dev), np.nan) for name in MODELS}
    rows, warning_log = [], []
    for fold, (tr, va) in enumerate(folds, 1):
        assert not np.intersect1d(dg[tr], dg[va]).size
        split.loc[dev_idx[va], "cv_fold"] = fold
        for name in MODELS:
            log(f"CV fold {fold}/5: {name}, train={len(tr)}, valid={len(va)}")
            _, p, seconds, notes = fit_predict(name, dev.iloc[tr], dev.iloc[va], SEED)
            oof[name][va] = p
            rows.append(dict(stage="cv", model=name, fold=fold, seconds=seconds,
                             train_n=len(tr), **score(dev.y.iloc[va], p, SEED)))
            warning_log.extend(notes)
    assert all(np.isfinite(p).all() for p in oof.values())
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/"metrics.csv", index=False)
    summary = metrics.groupby("model").agg(ap_mean=("ap", "mean"), ap_std=("ap", "std"),
                auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std"),
                top10_precision_mean=("top10_precision", "mean"),
                top10_lift_mean=("top10_lift", "mean"), seconds_sum=("seconds", "sum")).reset_index()
    summary.to_csv(out/"cv_summary.csv", index=False)
    means = summary.set_index("model").ap_mean
    selected = "hist_gbdt" if means["hist_gbdt"]-means["logistic"] >= 0.005 else "logistic"
    paired = metrics.pivot(index="fold", columns="model", values="ap")
    paired["ap_gain_gbdt_minus_logistic"] = paired.hist_gbdt-paired.logistic
    paired.to_csv(out/"paired_cv.csv")
    split.to_csv(out/"split_assignments.csv", index=False)
    oo = dev[["course_row_id", "y"]].copy()
    for name in MODELS:
        oo[name] = oof[name]
    oo.to_csv(out/"development_oof.csv", index=False)
    business = capacity_table(dev.y, oof, "development_oof", SEED)
    business.to_csv(out/"business_analysis.csv", index=False)
    audit = dict(full_rows=len(d), features=FEATURES, excluded=EXCLUDED,
                 full_positive_rate=float(d.y.mean()), full_positives=int(d.y.sum()),
                 development_rows=len(dev), test_rows=len(test), feature_profile_groups=int(len(np.unique(groups))),
                 repeated_allowed_profiles=int(d.duplicated(FEATURES).sum()),
                 repeated_full_nonid_features=int(d.duplicated([c for c in d if c not in ["y", "id", "course_row_id"]]).sum()),
                 cross_partition_group_overlap=0, customer_identity_verified=False,
                 missing_values=int(d.isna().sum().sum()), development_positive_rate=float(dev.y.mean()))
    dump(out/"data_audit.json", audit)
    pd.DataFrame([{"field": c, "used": c in FEATURES,
                   "reason": EXCLUDED.get(c, "联系前客户资料/历史，真实上线须核对快照"),
                   "missing": int(d[c].isna().sum()),
                   "unknown_category": int(d[c].eq("unknown").sum()) if d[c].dtype == "object" else 0}
                  for c in d]).to_csv(out/"feature_availability.csv", index=False)
    # Stability is development-only, never another look at the independent test.
    st_tr, st_va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED+1).split(dev, dev.y, dg))
    st_rows = []
    for name in MODELS:
        log(f"Development stability split: {name}")
        _, p, seconds, notes = fit_predict(name, dev.iloc[st_tr], dev.iloc[st_va], SEED+1)
        st_rows.append(dict(model=name, seed=SEED+1, split="development_group_holdout",
                            seconds=seconds, **score(dev.y.iloc[st_va], p, SEED)))
        warning_log.extend(notes)
    pd.DataFrame(st_rows).to_csv(out/"stability.csv", index=False)
    log("Running development-only leakage, calibration, importance and cost diagnostics")
    run_development_diagnostics(dev, folds, oof[selected], selected, out, SEED)
    diagnostic_hashes = {filename: sha(out/filename) for filename in DIAGNOSTIC_FILES}
    # Write the lock before either model sees test labels through any metric.
    locked = dict(locked_at=now(), selected_model=selected, seed=SEED, primary_metric="AP",
                  minimum_ap_gain_for_complex_model=0.005, capacity=0.1, tuning_trials_per_model=1,
                  features=FEATURES, excluded=EXCLUDED, model_parameters={
                      n: build_model(n, SEED).named_steps["model"].get_params() for n in MODELS},
                  source_hash=TRAIN_HASH, code_hashes=code_hashes(),
                  selection_evidence_sha256=sha(out/"cv_summary.csv"),
                  development_oof_sha256=sha(out/"development_oof.csv"),
                  split_assignments_sha256=sha(out/"split_assignments.csv"),
                  diagnostic_evidence_sha256=diagnostic_hashes)
    dump(out/"model_lock.json", locked)
    log(f"Locked model={selected}. Starting the single independent test evaluation stage")
    test_started = now()
    test_scores = {}
    for name in MODELS:
        model, p, seconds, notes = fit_predict(name, dev, test, SEED)
        test_scores[name] = p
        rows.append(dict(stage="independent_test", model=name, fold=0, seconds=seconds,
                         train_n=len(dev), **score(test.y, p, SEED)))
        warning_log.extend(notes)
        if name == selected:
            joblib.dump(model, out/"development_model.joblib")
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    pd.concat([business, capacity_table(test.y, test_scores, "independent_test", SEED)],
              ignore_index=True).to_csv(out/"business_analysis.csv", index=False)
    tp = test[["course_row_id", "y"] + FEATURES].copy()
    for name in MODELS:
        tp[name] = test_scores[name]
    tp.to_csv(out/"local_test_predictions.csv", index=False)
    p = test_scores[selected]
    threshold_table(test.y, p).to_csv(out/"threshold_analysis.csv", index=False)
    subgroup_table(test, p, SEED).to_csv(out/"subgroup_metrics.csv", index=False)
    chosen = top_mask(p, seed=SEED)
    error_types = np.select([chosen & (test.y.to_numpy() == 0), ~chosen & (test.y.to_numpy() == 1)],
                            ["FP_contact_no_subscription", "FN_missed_subscription"], default="correct")
    all_errors = tp.assign(selected_top10=chosen, error_type=error_types, probability=p)
    all_errors.groupby("error_type").size().rename("count").to_csv(out/"error_type_counts.csv")
    fp = all_errors[all_errors.error_type == "FP_contact_no_subscription"].sort_values("probability", ascending=False).head(10)
    fn = all_errors[all_errors.error_type == "FN_missed_subscription"].sort_values("probability", ascending=True).head(10)
    pd.concat([fp, fn]).to_csv(out/"error_cases.csv", index=False)
    if len(fp)+len(fn) < 10:
        raise ValueError("Fewer than ten errors: investigate before acceptance")
    dump(out/"test_audit.json", dict(locked_at=locked["locked_at"], evaluation_started=test_started,
         evaluation_completed=now(), evaluation_passes_in_this_run=1, models_reported=MODELS,
         selected_before_test=selected, changed_after_test=False,
         label_use="Fixed metrics and descriptive analysis only; no model/threshold selection",
         historical_reproduction_note=("The locked test calculation has been repeated in prior "
             "reproducibility audits; no test result changed any model, feature, capacity, "
             "threshold, or development-only diagnostic choice."),
         prediction_file_sha256=sha(out/"local_test_predictions.csv")))
    dump(out/"run_manifest.json", dict(status="local_complete_hidden_pending", started_at=started, updated_at=now(),
         seed=SEED, python=platform.python_version(), platform=platform.platform(),
         command=sys.argv, dependencies={n: importlib.metadata.version(n) for n in
             ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib", "joblib", "threadpoolctl"]},
         code_hashes=code_hashes(), data_hashes={"train": TRAIN_HASH}, warnings=sorted(set(warning_log)),
         selected_model=selected, hidden_labels_accessed=False))
    log("Local experiment and locked test analysis exported")


def finalize(d, data_dir, out):
    locked = json.loads((out/"model_lock.json").read_text())
    if locked["code_hashes"] != code_hashes():
        raise ValueError("Modeling code/protocol changed since lock; do not quietly reuse the test set")
    for filename, key in [("cv_summary.csv", "selection_evidence_sha256"),
                          ("development_oof.csv", "development_oof_sha256"),
                          ("split_assignments.csv", "split_assignments_sha256")]:
        if sha(out/filename) != locked[key]:
            raise ValueError(f"Locked evidence changed: {filename}")
    for filename, expected in locked.get("diagnostic_evidence_sha256", {}).items():
        if sha(out/filename) != expected:
            raise ValueError(f"Locked diagnostic evidence changed: {filename}")
    verify(data_dir/"student_hidden_features.parquet", HIDDEN_HASH)
    log("Retraining locked model on all 600000 labeled rows")
    model = build_model(locked["selected_model"], SEED)
    model.fit(d[FEATURES], d.y)
    # The first content read of hidden features happens only after the locked full refit.
    hidden = pd.read_parquet(data_dir/"student_hidden_features.parquet")
    if len(hidden) != 150000 or set(hidden.columns) != (set(d.columns)-{"y"}):
        raise ValueError("Hidden feature schema mismatch")
    if not hidden.course_row_id.is_unique or hidden.course_row_id.isna().any():
        raise ValueError("Invalid hidden IDs")
    if set(d.course_row_id) & set(hidden.course_row_id):
        raise ValueError("Train and hidden ID overlap")
    p = model.predict_proba(hidden[FEATURES])[:, 1]
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid prediction probabilities")
    pred = pd.DataFrame({"course_row_id": hidden.course_row_id, "prediction": p})
    pred.to_csv(out/"predictions.csv", index=False)
    joblib.dump(model, out/"final_model.joblib")
    manifest = json.loads((out/"run_manifest.json").read_text())
    manifest.update(status="complete", updated_at=now(), hidden_rows=len(hidden),
                    prediction_rows=len(pred), prediction_sha256=sha(out/"predictions.csv"),
                    hidden_feedback_used=False)
    manifest["data_hashes"]["hidden_features"] = HIDDEN_HASH
    dump(out/"run_manifest.json", manifest)
    log("150000 hidden probabilities exported; no hidden labels or feedback accessed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/bank-marketing")
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    parser.add_argument("--phase", choices=["all", "develop", "finalize"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        d = load_train(args.data_dir)
        if args.phase in ("all", "develop"):
            develop(d, args.output)
        if args.phase in ("all", "finalize"):
            finalize(d, args.data_dir, args.output)
    from src.reporting import render
    render(args.output, ROOT/"presentation", ROOT/"docs")
    log("Report, figures and standalone presentation generated")


if __name__ == "__main__":
    main()
