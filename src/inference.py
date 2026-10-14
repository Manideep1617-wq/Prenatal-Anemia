"""Clinical inference helpers shared by the screening app and tests."""

import numpy as np


def positive_class_probability(model, features):
    """Read the probability for class 1 without assuming its column position."""
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)

    if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
        raise ValueError("Model returned probabilities with an unexpected shape")
    if len(positive_columns) != 1 or probabilities.shape[1] != len(classes):
        raise ValueError("Model does not expose exactly one anemia class")

    probability = float(probabilities[0, positive_columns[0]])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Model returned an invalid anemia probability")
    return probability


def who_anemia_label(hemoglobin):
    """WHO pregnancy screening threshold used to label the training data."""
    if hemoglobin is None or not np.isfinite(hemoglobin):
        return None
    return "Anemic" if hemoglobin < 11.0 else "Non-Anemic"


def who_severity(hemoglobin):
    """Return the WHO hemoglobin severity band used by the app."""
    if hemoglobin is None or not np.isfinite(hemoglobin):
        return None
    if hemoglobin < 7.0:
        return "Severe Anemia"
    if hemoglobin < 10.0:
        return "Moderate Anemia"
    if hemoglobin < 11.0:
        return "Mild Anemia"
    return "Normal / Non-Anemic"


def modality_agreement(tabular_probability, image_probability):
    """Compare modality scores only when both trained-model scores exist."""
    if tabular_probability is None or image_probability is None:
        return None
    return max(
        0.0,
        min(100.0, (1.0 - abs(tabular_probability - image_probability)) * 100.0),
    )


def screening_assessment(hemoglobin, tabular_probability=None, image_probability=None):
    """Use WHO HGB for the final category; retain model-score agreement as context."""
    return {
        "label": who_anemia_label(hemoglobin),
        "severity": who_severity(hemoglobin),
        "modality_agreement": modality_agreement(tabular_probability, image_probability),
    }
