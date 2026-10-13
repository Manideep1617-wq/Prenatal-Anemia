import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import csv
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             precision_score, recall_score,
                             precision_recall_curve, roc_curve)
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')          # headless – no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import set_seed, load_config
from src.data_preprocess import load_tabular_data, get_image_loaders
from src.models import TabularModel, ImageModel


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_progress(step, total, msg=''):
    pct = int(step / total * 100)
    bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
    print(f"\r[{bar}] {pct}%  {msg}", flush=True)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def save_csv_rows(rows, headers, path):
    """Append or create a CSV file with the given rows."""
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(headers)
        for row in rows:
            w.writerow(row)


def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_training_curves(log_rows, metric_cols, title, save_path):
    """Plot training curves from a list of dicts."""
    epochs = [r['epoch'] for r in log_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in metric_cols:
        vals = [r[col] for r in log_rows]
        ax.plot(epochs, vals, marker='o', label=col)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_precision_recall(y_true, y_score, title, save_path):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 – Train Image Model (ResNet‑18)
# ──────────────────────────────────────────────────────────────────────────────

def train_image_model(model, train_loader, val_loader, device, epochs, lr,
                      ckpt_dir, records_dir, figures_dir):
    """Fine-tune ResNet-18 and save comprehensive results."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_path = os.path.join(ckpt_dir, 'image_model_best.pt')
    epoch_log = []

    print(f"\n{'='*60}")
    print("  PHASE 1 – Training Image Model (ResNet-18)")
    print(f"{'='*60}")

    for epoch in range(epochs):
        t0 = time.time()
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

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_preds_list, val_labels_list, val_proba_list = [], [], []
        val_loss_total, val_n = 0.0, 0
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                loss_v = criterion(outputs, labels.to(device))
                val_loss_total += loss_v.item() * labels.size(0)
                val_n += labels.size(0)
                proba = softmax(outputs)[:, 1].cpu().numpy()
                _, pred = torch.max(outputs, 1)
                val_preds_list.extend(pred.cpu().numpy())
                val_labels_list.extend(labels.numpy())
                val_proba_list.extend(proba)

        val_loss = val_loss_total / val_n
        val_acc = accuracy_score(val_labels_list, val_preds_list)
        val_f1  = f1_score(val_labels_list, val_preds_list, average='weighted', zero_division=0)
        val_prec = precision_score(val_labels_list, val_preds_list, average='weighted', zero_division=0)
        val_rec  = recall_score(val_labels_list, val_preds_list, average='weighted', zero_division=0)
        try:
            val_auc = float(roc_auc_score(val_labels_list, val_proba_list))
        except Exception:
            val_auc = None

        elapsed = time.time() - t0
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 4),
            'val_f1': round(val_f1, 4),
            'val_precision': round(val_prec, 4),
            'val_recall': round(val_rec, 4),
            'val_auc': round(val_auc, 4) if val_auc else None,
            'best_val_acc': round(max(best_acc, val_acc), 4),
            'lr': round(scheduler.get_last_lr()[0], 8),
            'epoch_time_sec': round(elapsed, 1),
        }
        epoch_log.append(epoch_record)

        print(f"\n  Epoch {epoch+1}/{epochs} | "
              f"Train-loss={train_loss:.4f} Train-acc={train_acc:.4f} | "
              f"Val-loss={val_loss:.4f} Val-acc={val_acc:.4f} | Val-F1={val_f1:.4f} | "
              f"Time={elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✔ Best image model saved  (val-acc={best_acc:.4f})")

    # ─── Save per‑epoch training log ─────────────────────────────────────────
    save_json(epoch_log, os.path.join(records_dir, 'image_epoch_log.json'))
    headers = list(epoch_log[0].keys())
    save_csv_rows([list(r.values()) for r in epoch_log], headers,
                  os.path.join(records_dir, 'image_epoch_log.csv'))

    # ─── Plot training curves ────────────────────────────────────────────────
    plot_training_curves(epoch_log,
                         ['train_acc', 'val_acc'],
                         'Image Model – Accuracy Curves',
                         os.path.join(figures_dir, 'image_accuracy_curves.png'))
    plot_training_curves(epoch_log,
                         ['train_loss', 'val_loss'],
                         'Image Model – Loss Curves',
                         os.path.join(figures_dir, 'image_loss_curves.png'))

    # ─── Final evaluation on best model ──────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    final_preds, final_labels, final_proba = [], [], []
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

    # Classification report
    cls_labels = ['Non-Anemic', 'Anemic']
    cls_report = classification_report(final_labels, final_preds,
                                       target_names=cls_labels,
                                       output_dict=True, zero_division=0)
    cls_report_text = classification_report(final_labels, final_preds,
                                            target_names=cls_labels, zero_division=0)
    with open(os.path.join(records_dir, 'image_classification_report.txt'), 'w') as f:
        f.write(cls_report_text)
    save_json(cls_report, os.path.join(records_dir, 'image_classification_report.json'))

    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    pd.DataFrame(cm, index=cls_labels, columns=cls_labels).to_csv(
        os.path.join(records_dir, 'image_confusion_matrix.csv'))
    plot_confusion_matrix(cm, cls_labels,
                          'Image Model – Confusion Matrix',
                          os.path.join(figures_dir, 'image_confusion_matrix.png'))

    # ROC curve
    try:
        auc_val = float(roc_auc_score(final_labels, final_proba))
        plot_roc_curve(final_labels, final_proba,
                       'Image Model – ROC Curve',
                       os.path.join(figures_dir, 'image_roc_curve.png'))
        plot_precision_recall(final_labels, final_proba,
                              'Image Model – Precision‑Recall Curve',
                              os.path.join(figures_dir, 'image_pr_curve.png'))
    except Exception:
        auc_val = None

    img_metrics = {
        'accuracy':  float(accuracy_score(final_labels, final_preds)),
        'f1':        float(f1_score(final_labels, final_preds, average='weighted', zero_division=0)),
        'precision': float(precision_score(final_labels, final_preds, average='weighted', zero_division=0)),
        'recall':    float(recall_score(final_labels, final_preds, average='weighted', zero_division=0)),
        'auc':       auc_val,
        'confusion_matrix': cm.tolist(),
        'classification_report': cls_report,
        'best_epoch': int(np.argmax([r['val_acc'] for r in epoch_log]) + 1),
        'total_epochs': epochs,
    }
    print(f"\n  Image Model Final → Acc={img_metrics['accuracy']:.4f}  "
          f"F1={img_metrics['f1']:.4f}  AUC={img_metrics['auc']}")
    return img_metrics, best_path


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 – Train Tabular Model (XGBoost)
# ──────────────────────────────────────────────────────────────────────────────

def train_tabular_model(X_tr, y_tr, X_val, y_val, feature_names,
                        ckpt_dir, records_dir, figures_dir):
    """Train XGBoost on tabular (CBC + maternal health) data & save everything."""
    print(f"\n{'='*60}")
    print("  PHASE 2 – Training Tabular Model (XGBoost)")
    print(f"{'='*60}")

    tab_model = TabularModel()
    tab_model.fit(X_tr, y_tr, X_val, y_val)

    # Predictions & probabilities
    preds = tab_model.predict(X_val)
    proba = tab_model.predict_proba(X_val)

    cls_labels = ['Non-Anemic', 'Anemic']
    cls_report = classification_report(y_val, preds, target_names=cls_labels,
                                       output_dict=True, zero_division=0)
    cls_report_text = classification_report(y_val, preds,
                                            target_names=cls_labels, zero_division=0)
    with open(os.path.join(records_dir, 'tabular_classification_report.txt'), 'w') as f:
        f.write(cls_report_text)
    save_json(cls_report, os.path.join(records_dir, 'tabular_classification_report.json'))

    # Confusion matrix
    cm = confusion_matrix(y_val, preds)
    pd.DataFrame(cm, index=cls_labels, columns=cls_labels).to_csv(
        os.path.join(records_dir, 'tabular_confusion_matrix.csv'))
    plot_confusion_matrix(cm, cls_labels,
                          'Tabular Model (XGBoost) – Confusion Matrix',
                          os.path.join(figures_dir, 'tabular_confusion_matrix.png'))

    # ROC & PR curves
    try:
        auc_val = float(roc_auc_score(y_val, proba))
        plot_roc_curve(y_val, proba,
                       'Tabular Model (XGBoost) – ROC Curve',
                       os.path.join(figures_dir, 'tabular_roc_curve.png'))
        plot_precision_recall(y_val, proba,
                              'Tabular Model (XGBoost) – Precision‑Recall Curve',
                              os.path.join(figures_dir, 'tabular_pr_curve.png'))
    except Exception:
        auc_val = None

    # Feature importance
    importance = tab_model.model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    fi_df.to_csv(os.path.join(records_dir, 'tabular_feature_importance.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.35)))
    fi_df_plot = fi_df.head(20)
    ax.barh(fi_df_plot['feature'], fi_df_plot['importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('XGBoost Feature Importance (Top 20)')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'tabular_feature_importance.png'), dpi=200)
    plt.close(fig)

    # Save model
    xgb_path = os.path.join(ckpt_dir, 'xgb_tabular.json')
    tab_model.save(xgb_path)
    print(f"  ✔ XGBoost model saved to {xgb_path}")

    tab_metrics = {
        'accuracy':  float(accuracy_score(y_val, preds)),
        'f1':        float(f1_score(y_val, preds, average='weighted', zero_division=0)),
        'precision': float(precision_score(y_val, preds, average='weighted', zero_division=0)),
        'recall':    float(recall_score(y_val, preds, average='weighted', zero_division=0)),
        'auc':       auc_val,
        'confusion_matrix': cm.tolist(),
        'classification_report': cls_report,
    }
    print(f"  Tabular Model → Acc={tab_metrics['accuracy']:.4f}  "
          f"F1={tab_metrics['f1']:.4f}  AUC={tab_metrics['auc']}")
    return tab_metrics


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 – Train Transformer‑Fusion Model
# ──────────────────────────────────────────────────────────────────────────────

def train_fusion_model(fusion_model, train_loader, val_loader,
                       X_tr_tensor, y_tr_tensor,
                       X_val_tensor, y_val_tensor,
                       device, epochs, lr, batch_size,
                       ckpt_dir, records_dir, figures_dir):
    """
    Train the multimodal transformer fusion model.
    For each image batch, we randomly sample matching tabular rows
    (since image & tabular datasets are different sizes / populations).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(fusion_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_path = os.path.join(ckpt_dir, 'fusion_transformer.pt')
    epoch_log = []
    n_tab_train = X_tr_tensor.size(0)
    n_tab_val   = X_val_tensor.size(0)

    print(f"\n{'='*60}")
    print("  PHASE 3 – Training Transformer‑Fusion Model")
    print(f"{'='*60}")

    for epoch in range(epochs):
        t0 = time.time()
        fusion_model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, (imgs, img_labels) in enumerate(train_loader):
            bs = imgs.size(0)
            # Randomly sample tabular rows to pair with this image batch
            tab_idx = torch.randint(0, n_tab_train, (bs,))
            tab_batch = X_tr_tensor[tab_idx].to(device)
            # Use image labels (binary anemia) as ground truth for fusion
            # (fusion model outputs 4 classes → map binary to severity buckets later,
            #  for now train on image labels which are 0/1)
            lbl_batch = img_labels.to(device)

            optimizer.zero_grad()
            outputs = fusion_model(tab_batch, imgs.to(device))
            loss = criterion(outputs, lbl_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bs
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbl_batch).sum().item()
            total += bs

        train_loss = running_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0
        scheduler.step()

        # Validation
        fusion_model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        val_preds_list, val_labels_list = [], []
        with torch.no_grad():
            for i, (imgs, img_labels) in enumerate(val_loader):
                bs = imgs.size(0)
                tab_idx = torch.randint(0, n_tab_val, (bs,))
                tab_batch = X_val_tensor[tab_idx].to(device)
                lbl_batch = img_labels.to(device)
                outputs = fusion_model(tab_batch, imgs.to(device))
                loss_v = criterion(outputs, lbl_batch)
                val_loss_sum += loss_v.item() * bs
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == lbl_batch).sum().item()
                val_total += bs
                val_preds_list.extend(preds.cpu().numpy())
                val_labels_list.extend(lbl_batch.cpu().numpy())

        val_loss = val_loss_sum / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_f1 = f1_score(val_labels_list, val_preds_list, average='weighted', zero_division=0)
        val_prec = precision_score(val_labels_list, val_preds_list, average='weighted', zero_division=0)
        val_rec  = recall_score(val_labels_list, val_preds_list, average='weighted', zero_division=0)

        elapsed = time.time() - t0
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 4),
            'val_f1': round(val_f1, 4),
            'val_precision': round(val_prec, 4),
            'val_recall': round(val_rec, 4),
            'best_val_acc': round(max(best_acc, val_acc), 4),
            'lr': round(scheduler.get_last_lr()[0], 8),
            'epoch_time_sec': round(elapsed, 1),
        }
        epoch_log.append(epoch_record)

        print(f"\n  Epoch {epoch+1}/{epochs} | "
              f"Train-loss={train_loss:.4f} Train-acc={train_acc:.4f} | "
              f"Val-loss={val_loss:.4f} Val-acc={val_acc:.4f} | Val-F1={val_f1:.4f} | "
              f"Time={elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(fusion_model.state_dict(), best_path)
            print(f"  ✔ Best fusion model saved  (val-acc={best_acc:.4f})")

    # ─── Save epoch log ─────────────────────────────────────────────────────
    save_json(epoch_log, os.path.join(records_dir, 'fusion_epoch_log.json'))
    headers = list(epoch_log[0].keys())
    save_csv_rows([list(r.values()) for r in epoch_log], headers,
                  os.path.join(records_dir, 'fusion_epoch_log.csv'))
    plot_training_curves(epoch_log,
                         ['train_acc', 'val_acc'],
                         'Fusion Model – Accuracy Curves',
                         os.path.join(figures_dir, 'fusion_accuracy_curves.png'))
    plot_training_curves(epoch_log,
                         ['train_loss', 'val_loss'],
                         'Fusion Model – Loss Curves',
                         os.path.join(figures_dir, 'fusion_loss_curves.png'))

    # ─── Final evaluation ────────────────────────────────────────────────────
    fusion_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    fusion_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, img_labels) in enumerate(val_loader):
            bs = imgs.size(0)
            tab_idx = torch.randint(0, n_tab_val, (bs,))
            tab_batch = X_val_tensor[tab_idx].to(device)
            lbl_batch = img_labels.to(device)
            outputs = fusion_model(tab_batch, imgs.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl_batch.cpu().numpy())

    # Classification report
    cls_labels = ['Non-Anemic', 'Anemic']
    cls_report = classification_report(all_labels, all_preds,
                                       target_names=cls_labels,
                                       output_dict=True, zero_division=0)
    cls_report_text = classification_report(all_labels, all_preds,
                                            target_names=cls_labels, zero_division=0)
    with open(os.path.join(records_dir, 'fusion_classification_report.txt'), 'w') as f:
        f.write(cls_report_text)
    save_json(cls_report, os.path.join(records_dir, 'fusion_classification_report.json'))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    pd.DataFrame(cm, index=cls_labels, columns=cls_labels).to_csv(
        os.path.join(records_dir, 'fusion_confusion_matrix.csv'))
    plot_confusion_matrix(cm, cls_labels,
                          'Fusion Model – Confusion Matrix',
                          os.path.join(figures_dir, 'fusion_confusion_matrix.png'))

    fusion_acc = float(accuracy_score(all_labels, all_preds))
    fusion_f1 = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    fusion_prec = float(precision_score(all_labels, all_preds, average='weighted', zero_division=0))
    fusion_rec = float(recall_score(all_labels, all_preds, average='weighted', zero_division=0))

    fusion_metrics = {
        'accuracy':  fusion_acc,
        'f1':        fusion_f1,
        'precision': fusion_prec,
        'recall':    fusion_rec,
        'confusion_matrix': cm.tolist(),
        'classification_report': cls_report,
        'best_epoch': int(np.argmax([r['val_acc'] for r in epoch_log]) + 1),
        'total_epochs': epochs,
    }
    print(f"\n  Fusion Model Final → Acc={fusion_acc:.4f}  F1={fusion_f1:.4f}")
    return fusion_metrics


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    config  = load_config('src/config.yaml')
    set_seed(config['training']['seed'])

    DATA_DIR    = config['paths']['data_dir']
    IMAGE_DIR   = config['paths']['image_dir']
    RESULTS_DIR = config['paths']['results_dir']
    EPOCHS      = config['training'].get('image_epochs', 10)
    BATCH_SIZE  = config['training']['batch_size']
    LR          = config['training']['lr']

    # Create all output directories
    ckpt_dir    = os.path.join(RESULTS_DIR, 'checkpoints')
    records_dir = os.path.join(RESULTS_DIR, 'records')
    figures_dir = os.path.join(RESULTS_DIR, 'figures')
    ensure_dirs(RESULTS_DIR, ckpt_dir, records_dir, figures_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")

    # ─── Load tabular data ─────────────────────────────────────────────────
    print("\nLoading tabular datasets …")
    X_tr, y_tr, X_val, y_val, feature_names = load_tabular_data(DATA_DIR)
    print(f"Features: {feature_names}")
    print_progress(10, 100, "Tabular data loaded")

    # Save dataset statistics for the paper
    dataset_stats = {
        'tabular': {
            'train_samples': int(len(X_tr)),
            'val_samples':   int(len(X_val)),
            'num_features':  int(X_tr.shape[1]),
            'feature_names': feature_names,
            'train_label_distribution': {
                'anemic': int(np.sum(y_tr == 1)),
                'non_anemic': int(np.sum(y_tr == 0)),
            },
            'val_label_distribution': {
                'anemic': int(np.sum(y_val == 1)),
                'non_anemic': int(np.sum(y_val == 0)),
            },
        }
    }

    # ─── Load image data ───────────────────────────────────────────────────
    print("\nLoading image datasets …")
    train_img_loader, val_img_loader = get_image_loaders(IMAGE_DIR, BATCH_SIZE)
    dataset_stats['image'] = {
        'train_samples': len(train_img_loader.dataset),
        'val_samples':   len(val_img_loader.dataset),
        'image_size':    '224x224',
        'augmentations': ['RandomHorizontalFlip', 'RandomRotation(10)',
                          'ColorJitter(0.2,0.2)', 'Normalize(ImageNet)'],
    }
    print_progress(20, 100, "Image data loaded")

    # ─── Phase 1: Train image model ────────────────────────────────────────
    img_model = ImageModel(num_classes=2, pretrained=True).to(device)
    img_metrics, img_model_path = train_image_model(
        img_model, train_img_loader, val_img_loader, device, EPOCHS, LR,
        ckpt_dir, records_dir, figures_dir
    )
    print_progress(50, 100, "Image model trained")

    # ─── Phase 2: Train tabular model ──────────────────────────────────────
    tab_metrics = train_tabular_model(
        X_tr, y_tr, X_val, y_val, feature_names,
        ckpt_dir, records_dir, figures_dir
    )
    print_progress(70, 100, "Tabular model trained")

    # ─── Phase 3: Train transformer‑fusion model ──────────────────────────
    from src.models import TransformerFusionModel
    tab_input_dim = X_tr.shape[1]
    X_tr_tensor  = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_tensor  = torch.tensor(y_tr, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Fusion model uses 2 classes (binary) to match the image labels
    fusion_model = TransformerFusionModel(tab_input_dim=tab_input_dim,
                                          num_classes=2).to(device)
    fusion_metrics = train_fusion_model(
        fusion_model, train_img_loader, val_img_loader,
        X_tr_tensor, y_tr_tensor,
        X_val_tensor, y_val_tensor,
        device, EPOCHS, LR, BATCH_SIZE,
        ckpt_dir, records_dir, figures_dir
    )
    print_progress(90, 100, "Fusion model trained")

    # ─── Save consolidated metrics ─────────────────────────────────────────
    total_time = time.time() - start_time
    all_metrics = {
        'timestamp': datetime.datetime.now().isoformat(),
        'device': str(device),
        'training_time_minutes': round(total_time / 60, 2),
        'hyperparameters': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealing',
            'weight_decay': 1e-4,
            'seed': config['training']['seed'],
        },
        'dataset_statistics': dataset_stats,
        'image_model':   img_metrics,
        'tabular_model': tab_metrics,
        'fusion_model':  fusion_metrics,
        'summary': {
            'image_accuracy':    img_metrics['accuracy'],
            'image_f1':          img_metrics['f1'],
            'image_auc':         img_metrics['auc'],
            'tabular_accuracy':  tab_metrics['accuracy'],
            'tabular_f1':        tab_metrics['f1'],
            'tabular_auc':       tab_metrics['auc'],
            'fusion_accuracy':   fusion_metrics['accuracy'],
            'fusion_f1':         fusion_metrics['f1'],
        },
    }
    save_json(all_metrics, os.path.join(RESULTS_DIR, 'metrics.json'))

    # ─── Create a comparison bar chart ─────────────────────────────────────
    models = ['Image\n(ResNet-18)', 'Tabular\n(XGBoost)', 'Fusion\n(Transformer)']
    acc_vals = [img_metrics['accuracy'], tab_metrics['accuracy'], fusion_metrics['accuracy']]
    f1_vals  = [img_metrics['f1'], tab_metrics['f1'], fusion_metrics['f1']]

    x = np.arange(len(models))
    width = 0.3
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, acc_vals, width, label='Accuracy', color='#2196F3')
    bars2 = ax.bar(x + width/2, f1_vals, width, label='F1 Score', color='#FF9800')
    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2., b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2., b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison – Accuracy & F1')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'model_comparison.png'), dpi=200)
    plt.close(fig)

    # ─── Final summary ─────────────────────────────────────────────────────
    print_progress(100, 100, "DONE")
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Image   → Acc={img_metrics['accuracy']:.4f}  F1={img_metrics['f1']:.4f}  AUC={img_metrics['auc']}")
    print(f"  Tabular → Acc={tab_metrics['accuracy']:.4f}  F1={tab_metrics['f1']:.4f}  AUC={tab_metrics['auc']}")
    print(f"  Fusion  → Acc={fusion_metrics['accuracy']:.4f}  F1={fusion_metrics['f1']:.4f}")
    print(f"  Total training time: {total_time/60:.1f} minutes")
    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"    ├── metrics.json          (consolidated metrics)")
    print(f"    ├── checkpoints/          (model weights)")
    print(f"    ├── records/              (per-epoch logs, classification reports, confusion matrices)")
    print(f"    └── figures/              (plots for the research paper)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
