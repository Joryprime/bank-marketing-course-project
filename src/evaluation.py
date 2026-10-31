"""Ranking, capacity and threshold metrics, with explicit denominators."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             confusion_matrix)


def top_mask(probabilities, fraction=0.1, seed=20261022):
    p = np.asarray(probabilities, dtype=float)
    if not len(p) or not 0 < fraction <= 1 or not np.isfinite(p).all():
        raise ValueError("Nonempty finite scores and capacity in (0, 1] required")
    k = int(np.ceil(fraction * len(p)))
    # Stable, seeded random tie breaking; never use a row ID or label.
    order = np.lexsort((np.random.default_rng(seed).random(len(p)), -p))
    mask = np.zeros(len(p), dtype=bool)
    mask[order[:k]] = True
    return mask


def score(y, p, seed=20261022):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    chosen = top_mask(p, seed=seed)
    rate = float(y.mean())
    both = len(np.unique(y)) == 2
    precision = float(y[chosen].mean())
    return dict(n=len(y), positives=int(y.sum()), positive_rate=rate,
                ap=float(average_precision_score(y, p)) if both else np.nan,
                roc_auc=float(roc_auc_score(y, p)) if both else np.nan,
                brier=float(brier_score_loss(y, p)),
                accuracy_05=float(((p >= 0.5) == y).mean()),
                all_negative_accuracy=1-rate,
                top10_contacts=int(chosen.sum()), top10_hits=int(y[chosen].sum()),
                top10_precision=precision, top10_lift=precision/rate if rate else np.nan,
                top10_recall=float(y[chosen].sum()/y.sum()) if y.sum() else np.nan)


def capacity_table(y, score_map, stage, seed):
    y = np.asarray(y)
    n, positives = len(y), int(y.sum())
    rows = []
    for fraction in (0.05, 0.1, 0.2):
        k = int(np.ceil(n*fraction))
        rate = float(y.mean())
        rows.append(dict(stage=stage, strategy="random_expected", capacity=fraction, n=n,
                         contacts=k, total_positives=positives, hits=k*rate,
                         precision=rate, recall=k/n, lift=1.0, count_type="expectation"))
        random_mask = top_mask(np.zeros(n), fraction, seed)
        choices = {"random_realized": random_mask}
        choices.update({name: top_mask(p, fraction, seed) for name, p in score_map.items()})
        for name, chosen in choices.items():
            hits = int(y[chosen].sum())
            rows.append(dict(stage=stage, strategy=name, capacity=fraction, n=n, contacts=k,
                             total_positives=positives, hits=hits, precision=hits/k,
                             recall=hits/positives if positives else np.nan,
                             lift=(hits/k)/rate if rate else np.nan, count_type="observed"))
    return pd.DataFrame(rows)


def threshold_table(y, p):
    rows = []
    for threshold in (0.3, 0.5):
        tn, fp, fn, tp = confusion_matrix(y, np.asarray(p) >= threshold, labels=[0, 1]).ravel()
        rows.append(dict(threshold=threshold, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                         precision=tp/(tp+fp) if tp+fp else np.nan,
                         recall=tp/(tp+fn) if tp+fn else np.nan,
                         contacts=int(tp+fp)))
    return pd.DataFrame(rows)


def subgroup_table(frame, p, seed):
    y = frame.y.to_numpy()
    selected = top_mask(p, seed=seed)
    age_groups = pd.cut(frame.age, [-np.inf, 30, 60, np.inf],
                        labels=["age<30", "30<=age<60", "age>=60"], right=False)
    rows = []
    for group in age_groups.cat.categories:
        member = (age_groups == group).to_numpy()
        if not member.any():
            continue
        m = score(y[member], np.asarray(p)[member], seed)
        hits = int(y[member & selected].sum())
        contacts = int((member & selected).sum())
        rows.append(dict(group=group, n=m["n"], positives=m["positives"], positive_rate=m["positive_rate"],
                         ap=m["ap"], roc_auc=m["roc_auc"], global_policy_contacts=contacts,
                         global_policy_hits=hits, selection_rate=contacts/member.sum(),
                         global_policy_tpr=hits/m["positives"] if m["positives"] else np.nan,
                         global_policy_ppv=hits/contacts if contacts else np.nan))
    return pd.DataFrame(rows)
