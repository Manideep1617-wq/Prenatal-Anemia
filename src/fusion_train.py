import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from src.models import TransformerFusionModel
from src.utils import set_seed
import json
import os

def train_fusion_model(tab_train, tab_val, img_train_loader, img_val_loader, device, epochs, lr, results_dir):
    """Train the transformer‑fusion multimodal model.
    Args:
        tab_train (np.ndarray): Training tabular features.
        tab_val (np.ndarray): Validation tabular features.
        img_train_loader (DataLoader): Image training loader (returns (imgs, labels)).
        img_val_loader (DataLoader): Image validation loader.
        device (torch.device): Device for training.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        results_dir (str): Directory to save model and metrics.
    Returns:
        dict: validation accuracy and f1, plus path to saved checkpoint.
    """
    # Convert tabular data to tensors
    X_tr_tensor = torch.tensor(tab_train, dtype=torch.float32).to(device)
    y_tr_tensor = torch.tensor(tab_train_labels, dtype=torch.long).to(device)  # placeholder, will be overridden by image labels
    X_val_tensor = torch.tensor(tab_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(tab_val_labels, dtype=torch.long).to(device)

    # Model
    tab_input_dim = tab_train.shape[1]
    model = TransformerFusionModel(tab_input_dim=tab_input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_path = os.path.join(results_dir, 'fusion_transformer.pt')
    print("\n" + "="*60)
    print("  TRAINING TRANSFORMER‑FUSION MODEL")
    print("="*60)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (imgs, labels) in enumerate(img_train_loader):
            batch_start = i * imgs.size(0)
            batch_end = batch_start + imgs.size(0)
            tab_batch = X_tr_tensor[batch_start:batch_end]
            lbl_batch = labels.to(device)
            optimizer.zero_grad()
            outputs = model(tab_batch, imgs.to(device))
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
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(img_val_loader):
                batch_start = i * imgs.size(0)
                batch_end = batch_start + imgs.size(0)
                tab_batch = X_val_tensor[batch_start:batch_end]
                lbl_batch = labels.to(device)
                outputs = model(tab_batch, imgs.to(device))
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == lbl_batch).sum().item()
                val_total += lbl_batch.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"\n  Epoch {epoch+1}/{epochs} | Train‑acc={train_acc:.4f} | Val‑acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✔ Best fusion model saved (val‑acc={best_acc:.4f})")
    # Load best model for final metrics
    model.load_state_dict(torch.load(best_path, map_location=device))
    # Compute final metrics on validation set
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(img_val_loader):
            batch_start = i * imgs.size(0)
            batch_end = batch_start + imgs.size(0)
            tab_batch = X_val_tensor[batch_start:batch_end]
            lbl_batch = labels.to(device)
            outputs = model(tab_batch, imgs.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl_batch.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"\n  Fusion Final → Acc={acc:.4f}  F1={f1:.4f}")
    return {'accuracy': acc, 'f1': f1}, best_path
