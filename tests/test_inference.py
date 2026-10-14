import unittest

import numpy as np

from src.inference import (
    modality_agreement,
    positive_class_probability,
    screening_assessment,
    who_anemia_label,
    who_severity,
)


class FakeClassifier:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, features):
        return self.probabilities


class InferenceTests(unittest.TestCase):
    def test_probability_uses_anemia_class_label_not_column_order(self):
        model = FakeClassifier([1, 0], [[0.81, 0.19]])
        self.assertEqual(positive_class_probability(model, [object()]), 0.81)

    def test_probability_rejects_invalid_model_output(self):
        model = FakeClassifier([0, 1], [[0.2, np.nan]])
        with self.assertRaises(ValueError):
            positive_class_probability(model, [object()])

    def test_who_anemia_cutoff(self):
        self.assertEqual(who_anemia_label(10.99), "Anemic")
        self.assertEqual(who_anemia_label(11.0), "Non-Anemic")

    def test_who_severity_boundaries(self):
        cases = [
            (6.99, "Severe Anemia"),
            (7.0, "Moderate Anemia"),
            (9.99, "Moderate Anemia"),
            (10.0, "Mild Anemia"),
            (10.99, "Mild Anemia"),
            (11.0, "Normal / Non-Anemic"),
        ]
        for hemoglobin, expected in cases:
            with self.subTest(hemoglobin=hemoglobin):
                self.assertEqual(who_severity(hemoglobin), expected)

    def test_agreement_requires_both_model_scores(self):
        self.assertIsNone(modality_agreement(0.6, None))
        self.assertAlmostEqual(modality_agreement(0.8, 0.2), 40.0)

    def test_image_scores_do_not_override_who_assessment(self):
        result = screening_assessment(6.8, tabular_probability=0.01, image_probability=0.99)
        self.assertEqual(result["label"], "Anemic")
        self.assertEqual(result["severity"], "Severe Anemia")
        self.assertAlmostEqual(result["modality_agreement"], 2.0)


if __name__ == "__main__":
    unittest.main()
