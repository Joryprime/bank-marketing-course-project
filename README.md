# Bank Marketing — leak-safe course project

This repository implements the third final project, **Binary Classification with a Bank Dataset**. It predicts subscription propensity before the current contact and turns probabilities into a fixed-capacity contact ranking.

## What is submitted

- Reproducible code with one entry point: `main.py`
- Auditable local validation and independent-test results in `results/`
- Hidden-feature probabilities in `results/predictions.csv`
- Chinese final report in `docs/final_report.md`
- Five-minute standalone HTML presentation in `presentation/index.html`
- Editable PowerPoint defense deck in `presentation/Bank-Marketing-Defense.pptx`
- AI usage and verification record in `ai_usage_log.md`
- Project self-study entry in `docs/self_study.md`
- Viva questions and answers in `docs/viva_qa.md`
- Seven-person role and code-defense guide in `docs/team_roles_and_defense.md`
- Final handoff checklist in `SUBMISSION_CHECKLIST.md`
- Course-requirement compliance audit in `docs/compliance_audit.md`
- Development-only leakage, calibration, importance and cost evidence in `docs/supplement_analysis.md`

This repository-ready package intentionally excludes raw Parquet data and fitted `.joblib` model binaries. The course data is public but is kept out of Git; see `data/README.md` for the source and local setup. CSV/JSON result evidence is retained. The large row-level development artifacts referenced by the model lock are stored as `.csv.gz` to keep individual files within GitHub's web upload limit; use `gzip -dk results/development_oof.csv.gz results/split_assignments.csv.gz` to restore the CSV files.

## Run

Use Python 3.12. Obtain the two course Parquet files using `data/README.md`, then run in a fresh output directory:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python main.py --phase all --output results/reproduced
```

The command above runs the complete pipeline in one invocation and recreates the main outputs from the course data. For a staged audit, run `--phase develop`, inspect the newly written lock, then run `--phase finalize` using the same output directory. Development refuses to overwrite an existing lock; use a new output directory for a new experiment.

The existing `results/` directory contains the archived, auditable run used by the report and presentation. The repository omits model binaries because they can be regenerated, and omits the raw course data because it should be fetched from the course release rather than copied into Git.

## Leakage policy

Only fields demonstrably available before the current contact are allowed. `duration`, current-campaign/last-contact fields, target, source ID and course row ID are excluded. Exact duplicate allowed-feature profiles remain in the same partition. Every preprocessing step is fitted within each training fold. Model family and capacity are selected from development CV and locked before a fixed independent-test pass in each reproduction. Historical locked reruns were used only to verify reproducibility and never changed the model, features, capacity, threshold or diagnostics. Hidden features are read only after lock verification and full-data refit.

The `duration` experiment is an explicitly labeled development-only timing-violation diagnostic. It quantifies how an unavailable post-contact field can inflate offline scores; it is never used by the submitted model. Illustrative cost thresholds are also selected only from development OOF predictions.

This repository never contains hidden labels and does not infer them from row order, source data, external joins, or leaderboard feedback. See `docs/protocol.md` and the generated audit JSON files for the complete contract.
