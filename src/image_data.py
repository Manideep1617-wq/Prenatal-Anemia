"""Image dataset grouping and patient-independent train/validation splitting."""

import os
import re
import hashlib

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


_PATIENT_ID = re.compile(r"(?:non[-_ ]?anemic|anemic)[-_ ]*(\d+)", re.IGNORECASE)
_COPY_SUFFIX = re.compile(r"\s*\(\d+\)$")


def image_group_id(path, label):
    """Return a class-scoped patient ID or a normalized image stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _PATIENT_ID.search(stem)
    if match:
        identity = f"patient-{match.group(1).lstrip('0') or '0'}"
    else:
        identity = _COPY_SUFFIX.sub("", stem).strip().casefold()
    return f"class-{int(label)}:{identity}"


def remove_exact_duplicate_images(paths, labels):
    """Remove byte-identical files and reject exact duplicates with conflicting labels."""
    if len(paths) != len(labels):
        raise ValueError("Image paths and labels must have equal lengths")

    seen = {}
    unique_paths = []
    unique_labels = []
    duplicate_count = 0
    for path, label in zip(paths, labels):
        try:
            with open(path, "rb") as image_file:
                digest = hashlib.sha256(image_file.read()).digest()
        except OSError as exc:
            raise RuntimeError(f"Could not read image file {path!r}: {exc}") from exc

        previous_label = seen.get(digest)
        if previous_label is not None:
            if previous_label != int(label):
                raise ValueError(
                    f"Byte-identical image {path!r} appears in both anemia classes; "
                    "resolve its label before training."
                )
            duplicate_count += 1
            continue

        seen[digest] = int(label)
        unique_paths.append(path)
        unique_labels.append(int(label))

    return unique_paths, unique_labels, duplicate_count


def split_patient_groups(paths, labels, val_ratio=0.2, seed=42):
    """Split images by patient and choose a near-stratified validation split."""
    if len(paths) != len(labels) or not paths:
        raise ValueError("Image paths and labels must be non-empty and have equal lengths")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray([
        image_group_id(path, label) for path, label in zip(paths, labels)
    ])
    group_labels = {}
    for group, label in zip(groups, labels):
        previous = group_labels.setdefault(group, int(label))
        if previous != int(label):
            raise ValueError(
                f"Patient group {group!r} has conflicting anemia labels; "
                "correct the dataset labels before training."
            )

    if len(group_labels) < 4 or len(np.unique(labels)) < 2:
        raise ValueError("At least four patient groups and both anemia labels are required")

    splitter = GroupShuffleSplit(n_splits=100, test_size=val_ratio, random_state=seed)
    overall_rate = float(labels.mean())
    best_split = None
    best_score = float("inf")
    for train_indices, val_indices in splitter.split(paths, labels, groups):
        if len(np.unique(labels[train_indices])) < 2 or len(np.unique(labels[val_indices])) < 2:
            continue
        size_error = abs(len(val_indices) / len(paths) - val_ratio)
        class_error = abs(float(labels[val_indices].mean()) - overall_rate)
        score = size_error + class_error
        if score < best_score:
            best_score = score
            best_split = train_indices, val_indices

    if best_split is None:
        raise ValueError(
            "Could not create a patient-independent train/validation split "
            "containing both anemia classes."
        )
    return best_split
