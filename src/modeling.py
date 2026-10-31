"""The feature allowlist and both fixed model configurations live here."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = ["age", "balance", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "poutcome"]
FEATURES = NUMERIC + CATEGORICAL
EXCLUDED = {
    "y": "目标，不作输入", "course_row_id": "课程行标识，仅对齐", "id": "来源行标识，非客户标识",
    "duration": "本次联系后时长", "campaign": "含本次的联系累计数，无事前快照",
    "contact": "最后联系渠道，无事前计划快照", "day": "最后联系日期，无事前计划快照",
    "month": "最后联系月份，无事前计划快照",
}
MODELS = ["logistic", "hist_gbdt"]


def build_hist_model(seed, numeric=None, categorical=None):
    """Build the fixed HistGBDT pipeline for an explicit diagnostic feature set."""
    numeric = list(NUMERIC if numeric is None else numeric)
    categorical = list(CATEGORICAL if categorical is None else categorical)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan)
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("encode", encoder)]), categorical),
    ], remainder="drop")
    estimator = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=5.0, early_stopping=False,
        categorical_features=[False] * len(numeric) + [True] * len(categorical),
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def build_model(name, seed):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if name == "logistic":
        numeric_steps.append(("scale", StandardScaler()))
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    else:
        return build_hist_model(seed)
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("encode", encoder)]), CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
