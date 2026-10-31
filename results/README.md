# Result files

This directory contains the archived result tables and figures used by the report and presentation. The primary submission output is `predictions.csv`, which contains `course_row_id` and the predicted positive-class probability for the 150,000 hidden-feature rows.

The required course evidence is retained: `metrics.csv`, `predictions.csv`, `error_cases.csv`, `stability.csv`, `business_analysis.csv`, and `figures/`. Additional CSV/JSON files support validation, leakage analysis, calibration, subgroup checks, and the recorded lock/audit hashes. The large row-level files `development_oof.csv` and `split_assignments.csv` are stored as `.csv.gz`; restore them with `gzip -dk development_oof.csv.gz split_assignments.csv.gz`.

The full Parquet inputs and fitted `.joblib` models are not included in this repository package. A fresh run from verified course data recreates outputs in a separate directory, for example `results/reproduced/`, using the single command in the root README.
