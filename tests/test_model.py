import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import FEATURE_NAMES, FEATURES, load_data, train  # noqa: E402


def test_features_exist_and_codes_match_survey_scales():
    X, y = load_data()
    assert list(X.columns) == FEATURE_NAMES
    assert set(y.unique()) == {0, 1}
    for feature, (_, answers) in FEATURES.items():
        observed = set(X[feature].dropna().astype(int))
        assert observed <= set(answers), f"{feature} has codes outside the survey scale"


def test_model_beats_chance_and_predicts_probabilities():
    model = train()
    assert model.metrics["roc_auc"] > 0.6
    assert model.metrics["balanced_accuracy"] > 0.55

    best = {f: min(FEATURES[f][1]) for f in FEATURE_NAMES}
    best.update(fruit_day=5, veggies_day=5, nutritional_check=5, eating_out=1)
    worst = {f: max(FEATURES[f][1]) for f in FEATURE_NAMES}
    worst.update(fruit_day=1, veggies_day=1, nutritional_check=1, eating_out=5)

    p_best, p_worst = model.predict_proba(best), model.predict_proba(worst)
    assert 0 <= p_worst < p_best <= 1
