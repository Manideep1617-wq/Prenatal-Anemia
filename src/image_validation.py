"""Validate that an image checkpoint matches patient-independent metrics."""

import hashlib
import json


VALIDATION_SPLIT = "patient-grouped; exact byte-identical duplicates removed"


def checkpoint_sha256(checkpoint_path):
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_checkpoint_validation_error(checkpoint_path, metrics_path):
    """Return why an image checkpoint is unvalidated, or None if metrics match it."""
    try:
        with open(metrics_path, encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
    except (OSError, json.JSONDecodeError) as exc:
        return f"matching training metrics are unavailable: {exc}"

    if not isinstance(metrics, dict):
        return "training metrics are not a JSON object"
    dataset_stats = metrics.get("dataset_statistics", {})
    if not isinstance(dataset_stats, dict):
        return "training metrics do not contain image dataset statistics"
    image_stats = dataset_stats.get("image", {})
    if not isinstance(image_stats, dict):
        return "training metrics do not contain image dataset statistics"
    if image_stats.get("image_site") != "eye":
        return "training metrics do not confirm eye-only image training"
    if image_stats.get("validation_split") != VALIDATION_SPLIT:
        return "training metrics do not document duplicate-free, patient-grouped validation"

    artifact = metrics.get("image_model_artifact", {})
    if not isinstance(artifact, dict):
        return "training metrics do not identify the image checkpoint"
    expected_digest = artifact.get("checkpoint_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        return "training metrics do not identify the image checkpoint"

    try:
        actual_digest = checkpoint_sha256(checkpoint_path)
    except OSError as exc:
        return f"image checkpoint could not be read: {exc}"

    if actual_digest != expected_digest:
        return "checkpoint does not match the checkpoint evaluated in the metrics"
    return None
