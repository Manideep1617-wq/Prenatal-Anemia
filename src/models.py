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

class TransformerFusionModel(nn.Module):
    """Transformer‑based multimodal fusion.
    * Tabular encoder → 64‑dim embedding (linear + ReLU).
    * Image encoder (ResNet‑18 backbone) → 512‑dim embedding.
    * Concatenate embeddings → sequence length 2, embed dim 512.
    * Pass through a small TransformerEncoder (2 layers, 4 heads).
    * Take the CLS token (first token) and feed to a classification head.
    The model outputs 4 severity classes (0‑None, 1‑Mild, 2‑Moderate, 3‑Severe).
    """
    def __init__(self, tab_input_dim, img_embed_dim=512, num_classes=4,
                 transformer_dim=256, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        # Tabular encoder
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Image encoder (ResNet‑18)
        weights = tv_models.ResNet18_Weights.DEFAULT
        resnet = tv_models.resnet18(weights=weights)
        resnet.fc = nn.Identity()
        self.img_backbone = resnet
        self.img_proj = nn.Sequential(
            nn.Linear(img_embed_dim, transformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(2, transformer_dim))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=transformer_dim*4,
                                                   dropout=dropout,
                                                   activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, num_classes),
        )

    def forward(self, tab_x, img_x):
        # Encode each modality
        tab_feat = self.tab_encoder(tab_x)                 # (B, 64)
        img_feat = self.img_backbone(img_x)               # (B, 512)
        img_feat = self.img_proj(img_feat)                # (B, transformer_dim)
        # Project tabular features to transformer_dim
        tab_proj = nn.Linear(tab_feat.size(1), img_feat.size(1)).to(tab_feat.device)
        tab_feat_proj = tab_proj(tab_feat)                # (B, transformer_dim)
        seq = torch.stack([tab_feat_proj, img_feat], dim=1)  # (B, 2, transformer_dim)
        seq = seq + self.pos_embedding                     # broadcast positional encodings
        seq = seq.transpose(0, 1)                         # (2, B, D) for transformer
        transformed = self.transformer(seq)               # (2, B, D)
        cls_token = transformed[0]                        # (B, D)
        out = self.classifier(cls_token)                  # (B, num_classes)
        return out
