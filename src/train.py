import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

from src.utils import set_seed, load_config
from src.data_preprocess import load_tabular_data, get_image_loaders
from src.models import TabularModel, ImageModel


def print_progress(step, total, msg=''):
    pct = int(step / total * 100)
    bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
    print(f"\r[{bar}] {pct}%  {msg}", flush=True)


def train_image_model(model, train_loader, val_loader, device, epochs, lr, results_dir):
    """Fine-tune the EfficientNet-B0 image model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_path = os.path.join(results_dir, 'image_model_best.pt')

    print(f"\n{'='*60}")
    print("  PHASE 1 – Training Image Model (EfficientNet-B0)")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            if i % 20 == 0:
                print_progress(epoch * len(train_loader) + i,
                               epochs * len(train_loader),
                               f"Epoch {epoch+1}/{epochs} train-loss={loss.item():.4f}")

        train_acc = correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                val_preds.extend(pred.cpu().numpy())
                val_labels_list.extend(labels.numpy())

        val_acc = accuracy_score(val_labels_list, val_preds)
        val_f1  = f1_score(val_labels_list, val_preds, average='weighted', zero_division=0)
        print(f"\n  Epoch {epoch+1}/{epochs} | "
              f"Train-acc={train_acc:.4f} | Val-acc={val_acc:.4f} | Val-F1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✔ Best image model saved  (val-acc={best_acc:.4f})")

    # Final evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    final_preds, final_labels = [], []
    final_proba = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            prob = softmax(out)[:, 1].cpu().numpy()
            _, pred = torch.max(out, 1)
            final_preds.extend(pred.cpu().numpy())
            final_labels.extend(labels.numpy())
            final_proba.extend(prob)

    img_metrics = {
        'accuracy' : accuracy_score(final_labels, final_preds),
        'f1'       : f1_score(final_labels, final_preds, average='weighted', zero_division=0),
        'auc'      : float(roc_auc_score(final_labels, final_proba)) if len(set(final_labels)) > 1 else None,
        'confusion_matrix': confusion_matrix(final_labels, final_preds).tolist(),
    }
    print(f"\n  Image Model Final → Acc={img_metrics['accuracy']:.4f}  "
          f"F1={img_metrics['f1']:.4f}  AUC={img_metrics['auc']}")
    return img_metrics, best_path


def train_tabular_model(X_tr, y_tr, X_val, y_val, results_dir):
    """Train XGBoost on tabular (CBC + maternal health) data."""
    print(f"\n{'='*60}")
    print("  PHASE 2 – Training Tabular Model (XGBoost)")
    print(f"{'='*60}")
    tab_model = TabularModel()
    tab_model.fit(X_tr, y_tr, X_val, y_val)
    tab_metrics = tab_model.evaluate(X_val, y_val)
    print(f"  Tabular Model → Acc={tab_metrics['accuracy']:.4f}  "
          f"F1={tab_metrics['f1']:.4f}  AUC={tab_metrics['auc']}")
    xgb_path = os.path.join(results_dir, 'xgb_tabular.json')
    tab_model.save(xgb_path)
    print(f"  ✔ XGBoost model saved to {xgb_path}")
    return tab_metrics


def main():
    config  = load_config('src/config.yaml')
    set_seed(config['training']['seed'])

    DATA_DIR    = config['paths']['data_dir']
    IMAGE_DIR   = config['paths']['image_dir']
    RESULTS_DIR = config['paths']['results_dir']
    os.makedirs(RESULTS_DIR, exist_ok=True)

    EPOCHS     = config['training'].get('image_epochs', 10)
    BATCH_SIZE = config['training']['batch_size']
    LR         = config['training']['lr']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ─── Load tabular data ─────────────────────────────────────────────────
    print("\nLoading tabular datasets …")
    X_tr, y_tr, X_val, y_val, feature_names = load_tabular_data(DATA_DIR)
    print(f"Features: {feature_names}")
    print_progress(10, 100, "Tabular data loaded")

    # ─── Load image data ───────────────────────────────────────────────────
    print("\nLoading image datasets …")
    train_img_loader, val_img_loader = get_image_loaders(IMAGE_DIR, BATCH_SIZE)
    print_progress(20, 100, "Image data loaded")

    # ─── Phase 1: Train image model ────────────────────────────────────────
    img_model = ImageModel(num_classes=2, pretrained=True).to(device)
    img_metrics, img_model_path = train_image_model(
        img_model, train_img_loader, val_img_loader, device, EPOCHS, LR, RESULTS_DIR
    )
    print_progress(60, 100, "Image model trained")

    # ─── Phase 2: Train tabular model ──────────────────────────────────────
    tab_metrics = train_tabular_model(X_tr, y_tr, X_val, y_val, RESULTS_DIR)
    print_progress(80, 100, "Tabular model trained")

    # ─── Train transformer‑fusion model ────────────────────────────────────────
    # Convert tabular data to tensors for fusion training
    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    from src.models import TransformerFusionModel
    tab_input_dim = X_tr.shape[1]
    fusion_model = TransformerFusionModel(tab_input_dim=tab_input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_fusion_acc = 0.0
    best_fusion_path = os.path.join(RESULTS_DIR, 'fusion_transformer.pt')
    print("\n" + "="*60)
    print("  PHASE 3 – Training Transformer‑Fusion Model")
    print("="*60)
    for epoch in range(EPOCHS):
        fusion_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (imgs, labels) in enumerate(train_img_loader):
            batch_start = i * BATCH_SIZE
            batch_end = batch_start + imgs.size(0)
            tab_batch = X_tr_tensor[batch_start:batch_end]
            lbl_batch = labels.to(device)
            optimizer.zero_grad()
            outputs = fusion_model(tab_batch, imgs.to(device))
            loss = criterion(outputs, lbl_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * lbl_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbl_batch).sum().item()
            total += lbl_batch.size(0)
        train_acc = correct / total if total > 0 else 0
        scheduler.step()
        # Validation
        fusion_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(val_img_loader):
                batch_start = i * BATCH_SIZE
                batch_end = batch_start + imgs.size(0)
                tab_batch = X_val_tensor[batch_start:batch_end]
                lbl_batch = labels.to(device)
                outputs = fusion_model(tab_batch, imgs.to(device))
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == lbl_batch).sum().item()
                val_total += lbl_batch.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"\n  Epoch {epoch+1}/{EPOCHS} | Train‑acc={train_acc:.4f} | Val‑acc={val_acc:.4f}")
        if val_acc > best_fusion_acc:
            best_fusion_acc = val_acc
            torch.save(fusion_model.state_dict(), best_fusion_path)
            print(f"  ✔ Best fusion model saved (val‑acc={best_fusion_acc:.4f})")
    # Load best fusion model for final evaluation
    fusion_model.load_state_dict(torch.load(best_fusion_path, map_location=device))
    fusion_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_img_loader):
            batch_start = i * BATCH_SIZE
            batch_end = batch_start + imgs.size(0)
            tab_batch = X_val_tensor[batch_start:batch_end]
            lbl_batch = labels.to(device)
            outputs = fusion_model(tab_batch, imgs.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl_batch.cpu().numpy())
    fusion_acc = accuracy_score(all_labels, all_preds)
    fusion_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"\n  Fusion Model Final → Acc={fusion_acc:.4f}  F1={fusion_f1:.4f}")

    # ─── Save all metrics ──────────────────────────────────────────────────
    all_metrics = {
        'image_model' : img_metrics,
        'tabular_model': tab_metrics,
        'fusion_model' : {'accuracy': fusion_acc, 'f1': fusion_f1},
        'summary': {
            'image_accuracy'   : img_metrics['accuracy'],
            'image_f1'         : img_metrics['f1'],
            'image_auc'        : img_metrics['auc'],
            'tabular_accuracy' : tab_metrics['accuracy'],
            'tabular_f1'       : tab_metrics['f1'],
            'tabular_auc'      : tab_metrics['auc'],
            'fusion_accuracy'  : fusion_acc,
            'fusion_f1'        : fusion_f1,
        }
    }
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)

    print_progress(100, 100, "DONE")
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Image  → Acc={img_metrics['accuracy']:.4f}  F1={img_metrics['f1']:.4f}  AUC={img_metrics['auc']}")
    print(f"  Tabular→ Acc={tab_metrics['accuracy']:.4f}  F1={tab_metrics['f1']:.4f}  AUC={tab_metrics['auc']}")
    print(f"  Fusion → Acc={fusion_acc:.4f}  F1={fusion_f1:.4f}")
    print(f"  Metrics saved → {metrics_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
