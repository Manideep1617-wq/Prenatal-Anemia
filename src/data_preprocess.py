import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.image_data import IMAGE_FOLDERS, remove_exact_duplicate_images, split_patient_groups

# ─────────────────────────────────────────────────────────────────────────────
# 1. TABULAR DATA – load and create anemia labels
# ─────────────────────────────────────────────────────────────────────────────

def load_tabular_data(data_dir):
    """
    Load all available tabular datasets, merge where possible,
    create a binary anemia label (1 = anemic, 0 = non-anemic),
    return (X_train, y_train, X_val, y_val, feature_names).
    """
    frames = []

    # ── CBC data (115 K rows, gold-standard HGB column) ──────────────────────
    cbc_path = os.path.join(data_dir, 'cbc_dataframe.csv')
    if os.path.exists(cbc_path):
        print("Loading CBC dataset …")
        cbc = pd.read_csv(cbc_path)
        cbc = cbc.dropna(how='all')
        # WHO threshold for pregnant women: HGB < 11 g/dL = anemic
        cbc = cbc[cbc['HGB'].notna()]
        cbc['label'] = (cbc['HGB'] < 11.0).astype(int)
        keep_cols = ['WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCHC', 'MCH', 'RDW', 'PLT', 'MPV', 'label']
        existing = [c for c in keep_cols if c in cbc.columns]
        cbc = cbc[existing].dropna()
        frames.append(cbc)
        print(f"  CBC: {len(cbc):,} rows | anemic={cbc['label'].sum():,}  non-anemic={(cbc['label']==0).sum():,}")

    # ── Maternal Health CSV (1205 rows) ───────────────────────────────────────
    mat_path = os.path.join(data_dir, 'Dataset - Updated.csv')
    if os.path.exists(mat_path):
        print("Loading Maternal Health CSV …")
        mat = pd.read_csv(mat_path)
        num_cols = ['Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI', 'Heart Rate']
        for c in num_cols:
            if c in mat.columns:
                mat[c] = pd.to_numeric(mat[c], errors='coerce')
        # Use Risk Level as label (High=1, Low=0)
        if 'Risk Level' in mat.columns:
            mat['label'] = (mat['Risk Level'].str.strip().str.lower() == 'high').astype(int)
            keep = [c for c in num_cols + ['label'] if c in mat.columns]
            mat = mat[keep].dropna()
            # Pad missing cols so all frames have same feature set later
            frames.append(mat)
            print(f"  Maternal: {len(mat):,} rows | high-risk={mat['label'].sum():,}")

    if not frames:
        raise RuntimeError("No tabular data found in data/")

    # ── Combine: align columns (fill missing with 0) ──────────────────────────
    all_cols = sorted(set(c for df in frames for c in df.columns if c != 'label'))
    combined_frames = []
    for df in frames:
        for col in all_cols:
            if col not in df.columns:
                df[col] = 0.0
        combined_frames.append(df[all_cols + ['label']])
    combined = pd.concat(combined_frames, ignore_index=True)
    combined[all_cols] = combined[all_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    X = combined[all_cols].values.astype(np.float32)
    y = combined['label'].values.astype(np.int64)

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Tabular split → train={len(X_tr):,}  val={len(X_val):,}  features={X.shape[1]}")
    return X_tr, y_tr, X_val, y_val, all_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE DATASET – labeled eye images
# ─────────────────────────────────────────────────────────────────────────────

class AnemiaImageDataset(Dataset):
    """
    Loads only eye images from Dataset_sample/ folders:
      Eye_Anemic / Eye_Non_Anemic
    Label: 1 = Anemic, 0 = Non-Anemic
    """
    def __init__(self, image_dir, split='train', val_ratio=0.2, seed=42, transform=None):
        self.transform = transform
        all_image_paths = []
        all_labels = []

        for folder in IMAGE_FOLDERS:
            folder_path = os.path.join(image_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            label = 1 if ('Anemic' in folder and 'Non' not in folder) else 0
            for fn in sorted(os.listdir(folder_path)):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_image_paths.append(os.path.join(folder_path, fn))
                    all_labels.append(label)

        if len(all_image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

        raw_image_count = len(all_image_paths)
        self.image_paths, self.labels, duplicate_count = remove_exact_duplicate_images(
            all_image_paths, all_labels
        )
        if len(set(self.labels)) < 2:
            raise ValueError("Image dataset must contain both anemia classes")

        dimensions = []
        for path in self.image_paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions.append(image.size)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Invalid training image {path!r}: {exc}") from exc

        self.dataset_audit = {
            'image_site': 'eye',
            'raw_images': raw_image_count,
            'unique_images': len(self.image_paths),
            'exact_duplicates_removed': duplicate_count,
            'median_width_px': float(np.median([width for width, _ in dimensions])),
            'median_height_px': float(np.median([height for _, height in dimensions])),
            'images_with_short_side_below_224px': sum(
                min(width, height) < 224 for width, height in dimensions
            ),
        }

        train_indices, val_indices = split_patient_groups(
            self.image_paths, self.labels, val_ratio=val_ratio, seed=seed
        )
        if split not in ('train', 'val'):
            raise ValueError("split must be 'train' or 'val'")
        selected = train_indices if split == 'train' else val_indices

        self.image_paths = [self.image_paths[i] for i in selected]
        self.labels = [self.labels[i] for i in selected]
        self.dataset_audit['split'] = split
        print(f"Image {split} set: {len(self.image_paths):,} images | anemic={sum(self.labels):,}  non-anemic={sum(1 for l in self.labels if l==0):,}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.image_paths[idx]) as image:
                img = image.convert('RGB')
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Could not read training image {self.image_paths[idx]!r}: {exc}"
            ) from exc
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_image_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def get_image_loaders(image_dir, batch_size=32):
    train_tf, val_tf = get_image_transforms()
    train_ds = AnemiaImageDataset(image_dir, split='train', transform=train_tf)
    val_ds   = AnemiaImageDataset(image_dir, split='val',   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
