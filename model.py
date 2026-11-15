"""Healthy-lifestyle model: training, evaluation and prediction.

Dataset: "Food choices" survey of 125 college students (food_choices.csv).
Every answer in the survey is stored as a numeric code, so the model works on
those codes directly. The scales below follow the survey's codebook.

Target: does the student describe their current diet as healthy / balanced?
(`diet_current_coded == 1`, a label assigned by the dataset authors from the
free-text answer "describe your current diet").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).with_name("food_choices.csv")
RANDOM_STATE = 42

# Feature -> (question, {code: answer}) taken from the survey codebook.
FEATURES: dict[str, tuple[str, dict[int, str]]] = {
    "exercise": (
        "How often do you exercise?",
        {1: "Every day", 2: "2–3 times a week", 3: "Once a week", 4: "Sometimes", 5: "Never"},
    ),
    "fruit_day": (
        "How likely are you to eat fruit in a regular day?",
        {1: "Very unlikely", 2: "Unlikely", 3: "Neutral", 4: "Likely", 5: "Very likely"},
    ),
    "veggies_day": (
        "How likely are you to eat vegetables in a regular day?",
        {1: "Very unlikely", 2: "Unlikely", 3: "Neutral", 4: "Likely", 5: "Very likely"},
    ),
    "cook": (
        "How often do you cook?",
        {
            1: "Every day",
            2: "A couple of times a week",
            3: "Whenever I can, not often",
            4: "Only help on holidays",
            5: "Never",
        },
    ),
    "eating_out": (
        "How often do you eat out in a typical week?",
        {1: "Never", 2: "1–2 times", 3: "2–3 times", 4: "3–5 times", 5: "Every day"},
    ),
    "sports": ("Do you play a sport?", {1: "Yes", 2: "No"}),
    "nutritional_check": (
        "Do you check nutritional values on food labels?",
        {
            1: "Never",
            2: "Only on certain products",
            3: "Very rarely",
            4: "On most products",
            5: "On everything",
        },
    ),
    "vitamins": ("Do you take vitamins?", {1: "Yes", 2: "No"}),
}
FEATURE_NAMES = list(FEATURES)
TARGET = "diet_current_coded"
HEALTHY_CODE = 1


@dataclass
class TrainedModel:
    pipeline: Pipeline
    metrics: dict[str, float]
    importances: dict[str, float]
    n_samples: int
    healthy_rate: float

    def predict_proba(self, answers: dict[str, int]) -> float:
        """Probability that this profile has a healthy / balanced diet."""
        row = pd.DataFrame([[answers[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        return float(self.pipeline.predict_proba(row)[0, 1])


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    y = (df[TARGET] == HEALTHY_CODE).astype(int)
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            # A few survey answers are blank: fill them with the most common answer
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Repeated stratified 5-fold cross-validation (the dataset is too small for one split)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    scores = cross_validate(
        build_pipeline(), X, y, cv=cv, scoring=["accuracy", "balanced_accuracy", "roc_auc"]
    )
    return {
        "accuracy": float(np.mean(scores["test_accuracy"])),
        "balanced_accuracy": float(np.mean(scores["test_balanced_accuracy"])),
        "roc_auc": float(np.mean(scores["test_roc_auc"])),
    }


def train(path: Path = DATA_PATH) -> TrainedModel:
    X, y = load_data(path)
    metrics = evaluate(X, y)
    pipeline = build_pipeline().fit(X, y)
    importances = dict(
        sorted(
            zip(FEATURE_NAMES, pipeline.named_steps["forest"].feature_importances_),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    return TrainedModel(pipeline, metrics, importances, len(y), float(y.mean()))


if __name__ == "__main__":
    model = train()
    print(f"Samples: {model.n_samples}  (healthy diet: {model.healthy_rate:.0%})")
    for name, value in model.metrics.items():
        print(f"{name:>18}: {value:.3f}")
    print("\nFeature importance:")
    for name, value in model.importances.items():
        print(f"  {name:<18} {value:.3f}")
