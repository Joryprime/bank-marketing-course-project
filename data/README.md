# Data contract

This project uses the public course release `bank-marketing` from the course data warehouse. Raw Parquet files are intentionally excluded from Git.

Course source: [fintech-ml-course-data](https://github.com/xqwang-sh/fintech-ml-course-data). The course project page identifies the release directory as `releases/projects/v1/data/`. Obtain the `bank-marketing` project files there and place the two Parquet files below this directory before running the project:

```text
data/bank-marketing/student_train.parquet
data/bank-marketing/student_hidden_features.parquet
```

- `student_train.parquet`: 600,000 labeled rows; expected SHA-256 `921e6430ae470959b5795c075736162e320d891bece827d905fe6f3483b9031a`.
- `student_hidden_features.parquet`: 150,000 unlabeled rows; expected SHA-256 `4169285aa7994d4b79bfb3894cbc0012fe2bc7fbc17a464d12d1b2aa7deacf6f`.
- Metadata and license notices are retained in `data/metadata/`.

The hidden feature file is not read until `results/model_lock.json` exists and all locked evidence hashes pass. No hidden label, ordering inference, external lookup, or evaluation feedback is used. The data follows a real public dataset structure but is course-generated and is not production data from a bank.
