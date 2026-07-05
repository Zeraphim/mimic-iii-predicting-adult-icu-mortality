from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "mimic_mortality_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ARTIFACT_DIR / "cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


AI_HEALTHCARE_ROOT = ROOT.parent
DATA_DIR = AI_HEALTHCARE_ROOT / "Datasets" / "mimiciii" / "1.4"
FEATURE_PATH = ARTIFACT_DIR / "adult_icu_first24_features.csv"
RESULTS_PATH = ARTIFACT_DIR / "model_results.json"


LAB_ITEMS = {
    50882: "bicarbonate",
    50902: "chloride",
    50912: "creatinine",
    50931: "glucose",
    50971: "potassium",
    50983: "sodium",
    51006: "bun",
    51221: "hematocrit",
    51222: "hemoglobin",
    51265: "platelets",
    51301: "wbc",
    50813: "lactate",
}


@dataclass(frozen=True)
class TutorialResults:
    features: pd.DataFrame
    metrics: pd.DataFrame
    logistic_top_features: pd.DataFrame
    rf_top_features: pd.DataFrame
    figure_paths: dict[str, Path]
    summary: dict[str, object]


def _read_csv_gz(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Expected MIMIC-III table is missing: {path}")
    return pd.read_csv(path, compression="gzip", **kwargs)


def build_adult_icu_cohort() -> pd.DataFrame:
    """Build one row per adult hospital admission with a first ICU stay."""

    admissions = _read_csv_gz(
        "ADMISSIONS",
        usecols=[
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "ADMISSION_TYPE",
            "INSURANCE",
            "ETHNICITY",
            "HOSPITAL_EXPIRE_FLAG",
        ],
        parse_dates=["ADMITTIME"],
    )
    patients = _read_csv_gz(
        "PATIENTS",
        usecols=["SUBJECT_ID", "GENDER", "DOB"],
        parse_dates=["DOB"],
    )
    icu = _read_csv_gz(
        "ICUSTAYS",
        usecols=[
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "FIRST_CAREUNIT",
            "INTIME",
        ],
        parse_dates=["INTIME"],
    )

    # Use the first ICU stay for each admission so each label is represented once.
    first_icu = (
        icu.sort_values(["HADM_ID", "INTIME"])
        .drop_duplicates("HADM_ID", keep="first")
        .query("FIRST_CAREUNIT != 'NICU'")
    )
    cohort = (
        first_icu.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner")
        .merge(patients, on="SUBJECT_ID", how="inner")
        .copy()
    )
    cohort["AGE"] = (cohort["INTIME"] - cohort["DOB"]).dt.days / 365.242
    cohort = cohort[cohort["AGE"] >= 18].copy()

    # MIMIC masks patients over 89 by shifting dates. Capping avoids false ages.
    cohort["AGE"] = cohort["AGE"].clip(upper=90)
    cohort["ETHNICITY_GROUP"] = cohort["ETHNICITY"].map(_collapse_ethnicity)

    return cohort[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "INTIME",
            "AGE",
            "GENDER",
            "ADMISSION_TYPE",
            "INSURANCE",
            "ETHNICITY_GROUP",
            "FIRST_CAREUNIT",
            "HOSPITAL_EXPIRE_FLAG",
        ]
    ].rename(columns={"HOSPITAL_EXPIRE_FLAG": "MORTALITY_LABEL"})


def _collapse_ethnicity(value: object) -> str:
    text = str(value).upper()
    if "WHITE" in text:
        return "WHITE"
    if "BLACK" in text:
        return "BLACK"
    if "HISPANIC" in text or "LATINO" in text:
        return "HISPANIC/LATINO"
    if "ASIAN" in text:
        return "ASIAN"
    if "UNKNOWN" in text or "UNABLE" in text or "DECLINED" in text:
        return "UNKNOWN"
    return "OTHER"


def aggregate_first24_labs(cohort: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selected lab values from the first 24 hours after ICU admission."""

    hadm_to_intime = cohort.set_index("HADM_ID")["INTIME"]
    selected = []
    usecols = ["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"]

    for chunk in pd.read_csv(
        DATA_DIR / "LABEVENTS.csv.gz",
        compression="gzip",
        usecols=usecols,
        chunksize=750_000,
    ):
        chunk = chunk.dropna(subset=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"])
        chunk["HADM_ID"] = chunk["HADM_ID"].astype("int64")
        chunk["ITEMID"] = chunk["ITEMID"].astype("int64")
        chunk = chunk[
            chunk["ITEMID"].isin(LAB_ITEMS)
            & chunk["HADM_ID"].isin(hadm_to_intime.index)
        ].copy()
        if chunk.empty:
            continue

        chunk["CHARTTIME"] = pd.to_datetime(chunk["CHARTTIME"], errors="coerce")
        chunk["INTIME"] = chunk["HADM_ID"].map(hadm_to_intime)
        hours_after_icu = (chunk["CHARTTIME"] - chunk["INTIME"]).dt.total_seconds() / 3600
        chunk = chunk[(hours_after_icu >= 0) & (hours_after_icu <= 24)].copy()
        if chunk.empty:
            continue

        chunk["LAB_NAME"] = chunk["ITEMID"].map(LAB_ITEMS)
        selected.append(chunk[["HADM_ID", "LAB_NAME", "VALUENUM"]])

    if not selected:
        return pd.DataFrame(index=cohort["HADM_ID"])

    labs = pd.concat(selected, ignore_index=True)
    aggregated = labs.groupby(["HADM_ID", "LAB_NAME"])["VALUENUM"].agg(
        ["min", "max", "mean", "count"]
    )
    wide = aggregated.unstack("LAB_NAME")
    wide.columns = [f"{lab}_{stat}" for stat, lab in wide.columns]
    return wide.reset_index()


def build_feature_table(force_rebuild: bool = False) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    if FEATURE_PATH.exists() and not force_rebuild:
        return pd.read_csv(FEATURE_PATH)

    cohort = build_adult_icu_cohort()
    labs = aggregate_first24_labs(cohort)
    features = cohort.merge(labs, on="HADM_ID", how="left")

    # Do not persist row-level MIMIC identifiers in the tutorial feature cache.
    features = features.drop(columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"])
    features.to_csv(FEATURE_PATH, index=False)
    return features


def _make_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )


def train_and_evaluate(features: pd.DataFrame) -> TutorialResults:
    warnings.filterwarnings("ignore", category=UserWarning)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    target = "MORTALITY_LABEL"
    excluded = {"SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", target}
    X = features.drop(columns=[col for col in excluded if col in features.columns])
    y = features[target].astype(int)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = _make_preprocessor(numeric_features, categorical_features)
    models = {
        "Logistic regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random forest": Pipeline(
            steps=[
                (
                    "preprocess",
                    _make_preprocessor(numeric_features, categorical_features),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        min_samples_leaf=25,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    metric_rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        hard_predictions = (probabilities >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, hard_predictions).ravel()
        metric_rows.append(
            {
                "model": name,
                "auroc": roc_auc_score(y_test, probabilities),
                "average_precision": average_precision_score(y_test, probabilities),
                "balanced_accuracy": balanced_accuracy_score(y_test, hard_predictions),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        )
        predictions[name] = probabilities

    metrics = pd.DataFrame(metric_rows)
    logistic_top = _logistic_coefficients(models["Logistic regression"])
    rf_top = _random_forest_importance(models["Random forest"], X_test, y_test)
    figures = _write_figures(y_test, predictions, metrics, logistic_top, rf_top)

    summary = {
        "cohort_rows": int(len(features)),
        "mortality_count": int(y.sum()),
        "mortality_rate": float(y.mean()),
        "feature_count": int(X.shape[1]),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "test_rows": int(len(y_test)),
        "metrics": metrics.to_dict(orient="records"),
        "lab_itemids": LAB_ITEMS,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    metrics.to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)
    logistic_top.to_csv(ARTIFACT_DIR / "logistic_top_features.csv", index=False)
    rf_top.to_csv(ARTIFACT_DIR / "rf_permutation_importance.csv", index=False)

    return TutorialResults(features, metrics, logistic_top, rf_top, figures, summary)


def _feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    return preprocessor.get_feature_names_out().tolist()


def _clean_feature_name(name: str) -> str:
    return (
        name.replace("numeric__", "")
        .replace("categorical__", "")
        .replace("_", " ")
        .title()
    )


def _logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    coefficients = model.named_steps["model"].coef_[0]
    rows = pd.DataFrame(
        {
            "feature": [_clean_feature_name(name) for name in _feature_names(model)],
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    return rows.sort_values("abs_coefficient", ascending=False).head(15)


def _random_forest_importance(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="average_precision",
        n_jobs=1,
    )
    rows = pd.DataFrame(
        {
            "feature": [_clean_feature_name(name) for name in X_test.columns],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return rows.sort_values("importance_mean", ascending=False).head(15)


def _write_figures(
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    logistic_top: pd.DataFrame,
    rf_top: pd.DataFrame,
) -> dict[str, Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, probabilities in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = metrics.loc[metrics["model"] == name, "auroc"].iloc[0]
        ax.plot(fpr, tpr, linewidth=2.2, label=f"{name} AUROC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", label="Chance")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    paths["roc"] = ARTIFACT_DIR / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(paths["roc"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    baseline = y_test.mean()
    for name, probabilities in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        ap = metrics.loc[metrics["model"] == name, "average_precision"].iloc[0]
        ax.plot(recall, precision, linewidth=2.2, label=f"{name} AP={ap:.3f}")
    ax.axhline(baseline, color="#94a3b8", linestyle="--", label=f"Baseline={baseline:.3f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    paths["pr"] = ARTIFACT_DIR / "precision_recall_curve.png"
    fig.tight_layout()
    fig.savefig(paths["pr"], dpi=180)
    plt.close(fig)

    best_model = metrics.sort_values("average_precision", ascending=False).iloc[0]["model"]
    hard_predictions = (predictions[best_model] >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        hard_predictions,
        display_labels=["Survived", "Died"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix: {best_model}")
    paths["confusion"] = ARTIFACT_DIR / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(paths["confusion"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    plot_df = logistic_top.sort_values("coefficient")
    colors = np.where(plot_df["coefficient"] > 0, "#b91c1c", "#0369a1")
    ax.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_title("Largest Logistic Regression Coefficients")
    ax.set_xlabel("Standardized coefficient")
    paths["logistic"] = ARTIFACT_DIR / "logistic_coefficients.png"
    fig.tight_layout()
    fig.savefig(paths["logistic"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    plot_df = rf_top.sort_values("importance_mean")
    ax.barh(plot_df["feature"], plot_df["importance_mean"], color="#0f766e")
    ax.set_title("Random Forest Permutation Importance")
    ax.set_xlabel("Mean average-precision decrease")
    paths["importance"] = ARTIFACT_DIR / "rf_permutation_importance.png"
    fig.tight_layout()
    fig.savefig(paths["importance"], dpi=180)
    plt.close(fig)

    return paths


def run_full_analysis(force_rebuild: bool = False) -> TutorialResults:
    features = build_feature_table(force_rebuild=force_rebuild)
    return train_and_evaluate(features)


if __name__ == "__main__":
    results = run_full_analysis()
    print(json.dumps(results.summary, indent=2))
