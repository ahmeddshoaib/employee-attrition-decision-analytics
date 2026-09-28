"""Comparable attrition models and segment decision support."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


TARGET = "Attrition"
IDENTIFIERS = ["EmployeeID"]


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError("Attrition column is required")
    frame[TARGET] = frame[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if frame[TARGET].isna().any():
        raise ValueError("Attrition must contain yes/no or 1/0")
    return frame


def preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    features = frame.drop(columns=[TARGET, *IDENTIFIERS], errors="ignore")
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ]
    )


def metric_row(name: str, y_true: pd.Series, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    return {
        "data_label": "synthetic_demo",
        "model": name,
        "n_test": len(y_true),
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
    }


def evaluate_models(frame: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame[TARGET],
        random_state=random_state,
    )
    x_train = train.drop(columns=[TARGET, *IDENTIFIERS], errors="ignore")
    x_test = test.drop(columns=[TARGET, *IDENTIFIERS], errors="ignore")
    y_train, y_test = train[TARGET], test[TARGET]

    prep = preprocessor(frame)
    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=random_state),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        pipeline = Pipeline([("prepare", prep), ("model", model)])
        fit_arguments = {}
        if name == "gradient_boosting":
            fit_arguments["model__sample_weight"] = compute_sample_weight("balanced", y_train)
        pipeline.fit(x_train, y_train, **fit_arguments)
        probability = pipeline.predict_proba(x_test)[:, 1]
        results.append(metric_row(name, y_test, probability))
        fitted[name] = pipeline

    best_name = max(results, key=lambda row: row["roc_auc"])["model"]
    best = fitted[best_name]
    probability = best.predict_proba(x_test)[:, 1]
    scored = test[[*IDENTIFIERS, TARGET]].copy()
    scored["attrition_probability"] = probability
    scored["selected_model"] = best_name
    return pd.DataFrame(results), scored


def retention_priority(frame: pd.DataFrame) -> pd.DataFrame:
    priority = (
        frame.groupby(["Department", "JobRole"], observed=True)[TARGET]
        .agg(headcount="size", leavers="sum", attrition_rate="mean")
        .reset_index()
    )
    rate_cut = priority["attrition_rate"].median()
    volume_cut = priority["leavers"].median()
    priority["priority"] = np.select(
        [
            (priority["attrition_rate"] >= rate_cut) & (priority["leavers"] >= volume_cut),
            (priority["attrition_rate"] >= rate_cut) | (priority["leavers"] >= volume_cut),
        ],
        ["Priority review", "Monitor"],
        default="Standard",
    )
    return priority.sort_values(["priority", "leavers"], ascending=[True, False])


def save_figures(metrics: pd.DataFrame, priority: pd.DataFrame, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    comparison = metrics.set_index("model")[["f1", "roc_auc"]].sort_values("roc_auc")
    ax = comparison.plot.barh(figsize=(9, 4.8), color=["#2563eb", "#7c3aed"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    ax.set_title("Synthetic validation only: attrition model comparison")
    plt.tight_layout()
    plt.savefig(directory / "model_comparison.png", dpi=180)
    plt.close()

    colours = {"Priority review": "#dc2626", "Monitor": "#f59e0b", "Standard": "#2563eb"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, subset in priority.groupby("priority"):
        ax.scatter(subset["leavers"], subset["attrition_rate"], s=np.maximum(subset["headcount"], 25), alpha=0.75, label=label, color=colours[label])
    ax.set_xlabel("Leavers")
    ax.set_ylabel("Attrition rate")
    ax.set_title("Synthetic validation only: retention priority matrix")
    ax.legend()
    plt.tight_layout()
    plt.savefig(directory / "retention_priority_matrix.png", dpi=180)
    plt.close()
