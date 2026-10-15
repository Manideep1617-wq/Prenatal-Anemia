import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.image_validation import (
    VALIDATION_SPLIT,
    checkpoint_sha256,
    image_checkpoint_validation_error,
)


class ImageCheckpointValidationTests(unittest.TestCase):
    def test_matching_checkpoint_and_patient_grouped_metrics_are_accepted(self):
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "image_model.pt"
            metrics = Path(folder) / "metrics.json"
            checkpoint.write_bytes(b"trained checkpoint")
            metrics.write_text(json.dumps({
                "dataset_statistics": {
                    "image": {"validation_split": VALIDATION_SPLIT}
                },
                "image_model_artifact": {
                    "checkpoint_sha256": checkpoint_sha256(checkpoint)
                },
            }), encoding="utf-8")

            self.assertIsNone(
                image_checkpoint_validation_error(checkpoint, metrics)
            )

    def test_old_random_split_metrics_are_rejected(self):
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "image_model.pt"
            metrics = Path(folder) / "metrics.json"
            checkpoint.write_bytes(b"old checkpoint")
            metrics.write_text(json.dumps({
                "dataset_statistics": {"image": {"train_samples": 100}}
            }), encoding="utf-8")

            error = image_checkpoint_validation_error(checkpoint, metrics)

        self.assertIn("patient-grouped validation", error)

    def test_changed_checkpoint_does_not_match_saved_validation(self):
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "image_model.pt"
            metrics = Path(folder) / "metrics.json"
            checkpoint.write_bytes(b"evaluated checkpoint")
            metrics.write_text(json.dumps({
                "dataset_statistics": {
                    "image": {"validation_split": VALIDATION_SPLIT}
                },
                "image_model_artifact": {
                    "checkpoint_sha256": checkpoint_sha256(checkpoint)
                },
            }), encoding="utf-8")
            checkpoint.write_bytes(b"replaced checkpoint")

            error = image_checkpoint_validation_error(checkpoint, metrics)

        self.assertIn("does not match", error)


if __name__ == "__main__":
    unittest.main()
