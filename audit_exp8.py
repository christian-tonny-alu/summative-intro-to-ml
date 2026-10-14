"""Independent reproducibility and decision-value audit for Experiment 8.

This intentionally depends only on NumPy, pandas, and scikit-learn so the strongest
reported result can be checked without rerunning the TensorFlow experiments.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


SEED = 42
EARLY_DAYS = 30
CACHE = Path("oulad_early30.csv")
OUTPUT = Path("artifacts/exp8_reproducibility_audit.json")
EXPECTED_AUC = 0.8273
MODEL_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "max_depth": 12,
    "max_features": 0.5,
    "min_samples_leaf": 4,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Mirror the notebook's feature engineering exactly."""
    out = df.copy()
    click_cols = (
        ["total_clicks"]
        + [f"week{i}_clicks" for i in range(1, 5)]
        + [column for column in out.columns if column.startswith("clicks_")]
    )

    out["no_vle_activity"] = out["total_clicks"].isna().astype(int)
    out[click_cols] = out[click_cols].fillna(0)
    out["click_trend"] = out["click_trend"].fillna(0)
    out["active_days"] = out["active_days"].fillna(0)
    out["distinct_materials"] = out["distinct_materials"].fillna(0)
    out["first_access"] = out["first_access"].fillna(EARLY_DAYS)

    out["early_n_assess"] = out["early_n_assess"].fillna(0)
    out["early_score_mean"] = out["early_score_mean"].fillna(
        out["early_score_mean"].median()
    )
    out["early_submit_lead"] = out["early_submit_lead"].fillna(
        out["early_submit_lead"].median()
    )
    out["registered_early_by"] = -out["date_registration"].fillna(0)

    imd_order = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]
    out["imd_missing"] = out["imd_band"].isna().astype(int)
    out["imd_band_ord"] = out["imd_band"].map(
        {band: index for index, band in enumerate(imd_order)}
    )
    out["imd_band_ord"] = out["imd_band_ord"].fillna(out["imd_band_ord"].median())
    out["education_ord"] = out["highest_education"].map(
        {
            "No Formal quals": 0,
            "Lower Than A Level": 1,
            "A Level or Equivalent": 2,
            "HE Qualification": 3,
            "Post Graduate Qualification": 4,
        }
    )
    out["age_ord"] = out["age_band"].map({"0-35": 0, "35-55": 1, "55<=": 2})
    out["is_stem"] = out["code_module"].isin(["CCC", "DDD", "EEE", "FFF"]).astype(int)

    for column in click_cols:
        out[column] = np.log1p(out[column])

    one_hot = pd.get_dummies(
        out[["gender", "region", "disability", "code_module"]], drop_first=True
    ).astype(int)
    behaviour = click_cols + [
        "click_trend",
        "active_days",
        "distinct_materials",
        "first_access",
        "no_vle_activity",
        "registered_early_by",
        "early_score_mean",
        "early_n_assess",
        "early_submit_lead",
    ]
    context = [
        "imd_band_ord",
        "imd_missing",
        "education_ord",
        "age_ord",
        "is_stem",
        "num_of_prev_attempts",
        "studied_credits",
    ]
    features = pd.concat(
        [out[behaviour + context].reset_index(drop=True), one_hot.reset_index(drop=True)],
        axis=1,
    )
    assessment_columns = [
        "early_score_mean",
        "early_n_assess",
        "early_submit_lead",
    ]
    return features, assessment_columns


def fit_and_score(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float | int], np.ndarray]:
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[
        :, list(model.classes_).index("Distinction")
    ]
    y_true = (np.asarray(y_test) == "Distinction").astype(int)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics = {
        "rows": int(len(y_test)),
        "positives": int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 4),
        "all_rest_baseline_accuracy": round(float(1 - y_true.mean()), 4),
        "auc": round(float(roc_auc_score(y_true, probability)), 4),
        "accuracy": round(float(accuracy_score(y_true, prediction)), 4),
        "macro_f1": round(float(f1_score(y_true, prediction, average="macro")), 4),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    return metrics, probability


def overlap_summary(groups: pd.Series, splits: dict[str, np.ndarray]) -> dict[str, dict]:
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    result = {}
    for left, right in pairs:
        shared = set(groups.iloc[splits[left]]) & set(groups.iloc[splits[right]])
        result[f"{left}_vs_{right}"] = {
            "shared_students": len(shared),
            f"{right}_records_from_shared_students": int(
                groups.iloc[splits[right]].isin(shared).sum()
            ),
        }
    return result


def capacity_table(y_true: pd.Series, probability: np.ndarray) -> list[dict]:
    rows = []
    for share in (0.05, 0.10, 0.15, 0.20):
        review_count = max(1, int(np.ceil(len(probability) * share)))
        reviewed = np.argsort(-probability)[:review_count]
        found = int(y_true.iloc[reviewed].sum())
        rows.append(
            {
                "review_share": share,
                "students_reviewed": review_count,
                "distinctions_found": found,
                "precision_in_reviewed_group": round(found / review_count, 4),
                "recall_of_all_distinctions": round(found / int(y_true.sum()), 4),
            }
        )
    return rows


def subgroup_table(
    audit_frame: pd.DataFrame, y_true: pd.Series, probability: np.ndarray
) -> list[dict]:
    frame = audit_frame.copy()
    frame["target"] = y_true.to_numpy()
    frame["probability"] = probability
    frame["prediction"] = (probability >= 0.5).astype(int)
    frame["imd_group"] = pd.cut(
        frame["imd_band"].str.extract(r"^(\d+)")[0].astype(float),
        bins=[-1, 29, 69, 100],
        labels=["more_deprived_0_30", "middle_30_70", "less_deprived_70_100"],
    ).astype("object")
    frame["imd_group"] = frame["imd_group"].fillna("unknown")

    rows = []
    for column in ("gender", "disability", "imd_group"):
        for value, group in frame.groupby(column, dropna=False):
            positives = int(group["target"].sum())
            negatives = int(len(group) - positives)
            row = {
                "attribute": column,
                "group": str(value),
                "rows": int(len(group)),
                "positives": positives,
                "positive_rate": round(float(group["target"].mean()), 4),
                "selection_rate": round(float(group["prediction"].mean()), 4),
                "precision": (
                    round(
                        float(
                            precision_score(
                                group["target"],
                                group["prediction"],
                                zero_division=0,
                            )
                        ),
                        4,
                    )
                    if group["prediction"].sum()
                    else None
                ),
                "recall": (
                    round(float(recall_score(group["target"], group["prediction"])), 4)
                    if positives
                    else None
                ),
                "false_positive_rate": (
                    round(
                        float(
                            ((group["target"] == 0) & (group["prediction"] == 1)).sum()
                            / negatives
                        ),
                        4,
                    )
                    if negatives
                    else None
                ),
                "auc": (
                    round(float(roc_auc_score(group["target"], group["probability"])), 4)
                    if positives and negatives
                    else None
                ),
            }
            rows.append(row)
    return rows


def main() -> None:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} is required. Run the notebook's data-building cells first."
        )

    raw = pd.read_csv(CACHE)
    features, assessment_columns = engineer_features(raw)
    multiclass_target = raw["final_result"].reset_index(drop=True)
    binary_target = multiclass_target.where(
        multiclass_target == "Distinction", "Rest"
    )
    is_distinction = binary_target == "Distinction"
    groups = raw["id_student"].reset_index(drop=True)

    all_indices = np.arange(len(raw))
    train_indices, temporary_indices = train_test_split(
        all_indices,
        test_size=0.30,
        stratify=multiclass_target,
        random_state=SEED,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        stratify=multiclass_target.iloc[temporary_indices],
        random_state=SEED,
    )
    splits = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    train_validation_indices = np.concatenate([train_indices, validation_indices])

    reproduced, probability = fit_and_score(
        features.iloc[train_validation_indices],
        binary_target.iloc[train_validation_indices],
        features.iloc[test_indices],
        binary_target.iloc[test_indices],
    )
    reproduced["published_auc"] = EXPECTED_AUC
    reproduced["auc_matches_to_4dp"] = reproduced["auc"] == EXPECTED_AUC

    no_assessment_features = features.drop(columns=assessment_columns)
    no_assessment, _ = fit_and_score(
        no_assessment_features.iloc[train_validation_indices],
        binary_target.iloc[train_validation_indices],
        no_assessment_features.iloc[test_indices],
        binary_target.iloc[test_indices],
    )

    group_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    group_splits = list(
        group_cv.split(features, multiclass_target, groups=groups)
    )
    overall_rate = float(is_distinction.mean())
    group_train, group_test = min(
        group_splits,
        key=lambda split: abs(
            float(is_distinction.iloc[split[1]].mean()) - overall_rate
        ),
    )
    grouped, _ = fit_and_score(
        features.iloc[group_train],
        binary_target.iloc[group_train],
        features.iloc[group_test],
        binary_target.iloc[group_test],
    )
    grouped["student_overlap"] = len(
        set(groups.iloc[group_train]) & set(groups.iloc[group_test])
    )

    grouped_no_assessment, _ = fit_and_score(
        no_assessment_features.iloc[group_train],
        binary_target.iloc[group_train],
        no_assessment_features.iloc[group_test],
        binary_target.iloc[group_test],
    )
    grouped_no_assessment["student_overlap"] = 0

    test_audit_frame = raw.iloc[test_indices][
        ["gender", "disability", "imd_band"]
    ].reset_index(drop=True)
    test_target = is_distinction.iloc[test_indices].astype(int).reset_index(drop=True)
    output = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "seed": SEED,
            "cache_sha256": sha256(CACHE),
        },
        "data": {
            "rows": len(raw),
            "unique_students": int(groups.nunique()),
            "distinction_rows": int(is_distinction.sum()),
            "feature_count": features.shape[1],
        },
        "record_split_student_overlap": overlap_summary(groups, splits),
        "published_split_reproduction": reproduced,
        "published_split_without_early_assessment_features": no_assessment,
        "unseen_student_holdout": grouped,
        "unseen_student_holdout_without_early_assessment_features": grouped_no_assessment,
        "review_capacity_on_published_test_split": capacity_table(test_target, probability),
        "subgroup_diagnostics_at_threshold_0_5": subgroup_table(
            test_audit_frame, test_target, probability
        ),
        "interpretation_notes": [
            "Subgroup metrics are diagnostics, not proof of fairness or unfairness.",
            "The review-capacity table is exploratory because capacities are compared on the test split.",
            "Early assessment score may not have been available to a teacher by day 30; the ablation bounds the engagement-only claim.",
            "The grouped holdout tests unseen students but is one deterministic fold, not a final external validation.",
            "The notebook computes median imputations and one-hot columns before splitting; this is minor preprocessing leakage even though the target is not used.",
            "AUC measures ranking, not probability calibration; deployment thresholds require calibration on fresh validation data.",
        ],
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")

    print(f"Published split AUC: {reproduced['auc']:.4f} (reported {EXPECTED_AUC:.4f})")
    print(f"Without early assessments: {no_assessment['auc']:.4f}")
    print(f"Unseen-student holdout AUC: {grouped['auc']:.4f}")
    print(f"Unseen-student, no assessments: {grouped_no_assessment['auc']:.4f}")
    print(f"Wrote {OUTPUT}")
    if not reproduced["auc_matches_to_4dp"]:
        raise SystemExit("Published Experiment 8 AUC did not reproduce to four decimals.")


if __name__ == "__main__":
    main()
