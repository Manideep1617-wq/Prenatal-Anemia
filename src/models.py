import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torchvision.models as tv_models


# ─────────────────────────────────────────────────────────────────────────────
# 1. TABULAR MODEL – XGBoost classifier
# ─────────────────────────────────────────────────────────────────────────────

class TabularModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            verbosity=0,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        preds = self.predict(X)
        proba = self.predict_proba(X)
        acc  = accuracy_score(y, preds)
        f1   = f1_score(y, preds, average='weighted', zero_division=0)
        try:
            auc = float(roc_auc_score(y, proba))
        except Exception:
            auc = None
        return {'accuracy': float(acc), 'f1': float(f1), 'auc': auc}

    def save(self, path):
        self.model.save_model(path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE MODEL – ResNet-18 with ImageNet pretrained weights (torchvision)
#    No HuggingFace Hub required.
# ─────────────────────────────────────────────────────────────────────────────

class ImageModel(nn.Module):
    """
    Fine-tuned ResNet-18 for binary anemia detection from eye/nail images.
    Uses torchvision pretrained ImageNet weights — no internet/HuggingFace required.
    ResNet-18 embed_dim = 512.
    """
    EMBED_DIM = 512

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        # Replace final FC with our classifier
        in_feat = backbone.fc.in_features          # 512
        backbone.fc = nn.Linear(in_feat, num_classes)
        self.backbone = backbone
        print(f"  ImageModel: ResNet-18 (pretrained={pretrained})  embed_dim={in_feat}")

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        """Return 512-d feature vector (before classification head)."""
        feat = x
        for name, layer in self.backbone.named_children():
            if name == 'fc':
                break
            feat = layer(feat)
        return feat.flatten(1)   # (B, 512)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FUSION MODEL – late fusion of tabular + image embeddings
# ─────────────────────────────────────────────────────────────────────────────

class FusionModel(nn.Module):
    """
    Late-fusion model:
      tabular_branch  → 64-d vector
      ResNet-18       → 512-d vector
      Concatenate     → 576-d
      Classifier head → num_classes
    """
    def __init__(self, tab_input_dim, img_embed_dim=512, hidden_dim=256,
                 num_classes=2, dropout=0.3):
        super().__init__()

        # Tabular branch
        self.tab_branch = nn.Sequential(
            nn.Linear(tab_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Image branch (share weights with ImageModel)
        weights = tv_models.ResNet18_Weights.DEFAULT
        resnet = tv_models.resnet18(weights=weights)
        resnet.fc = nn.Identity()            # strip classifier head
        self.img_backbone = resnet
        self.img_proj = nn.Sequential(
            nn.Linear(img_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64 + 128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tab_x, img_x):
        tab_feat = self.tab_branch(tab_x)
        img_raw  = self.img_backbone(img_x)   # (B, 512)
        img_feat = self.img_proj(img_raw)
        combined = torch.cat([tab_feat, img_feat], dim=1)
        return self.classifier(combined)
