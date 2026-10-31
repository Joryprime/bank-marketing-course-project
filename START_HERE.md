# Bank Marketing final integrated package

This is the combined submission package. The locked model, source data, and original core results are unchanged. New material strengthens the development-only leakage analysis, cost sensitivity, coarse calibration check, feature diagnostics, and defense evidence.

## Start here

1. Read `docs/final_report.md` for the problem, validation, results, recommendations, and limitations.
2. Open `presentation/Bank-Marketing-Defense.pptx`. Slides 1–6 are the approximately five-minute main presentation; slides 7–10 are defense appendices.
3. Replace the `成员1–成员7` placeholders with real names and verify the division against actual contributions.
4. Use `docs/team_roles_and_defense.md` and `docs/viva_qa.md` to rehearse code inputs, outputs, leakage, test-set discipline, Lift, and live modifications.
5. `presentation/index.html` remains available as the standalone six-slide HTML version.
6. Read `docs/supplement_analysis.md`; all supplementary analyses use development data only.
7. To reproduce, follow the single command in `README.md` and specify a fresh output directory.

## Student actions still required

- Fill in real names, student IDs, class, and team contributions in the course submission form. This package does not invent personal information.
- Confirm the submission portal, repository visibility, and required prediction filename with the instructor.
- Do not upload `data/**/*.parquet`, `.venv/`, or model binaries to a public Git repository. The local archive includes the course data only to make reproduction self-contained.
- Each team member must be able to explain and modify the code during the defense.

## Evidence boundaries

`duration` appears only in a development-set leakage counterfactual; the final model uses the 11 contact-time features. Cost weights are illustrative, calibration bins do not guarantee production calibration, and Lift 4.09 is ranking enrichment rather than causal sales impact. Hidden labels were never obtained or used.
