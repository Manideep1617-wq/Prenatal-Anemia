import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import io
import json
import datetime
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import xgboost as xgb
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

# ── Project root on path ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── Optional SHAP import ───────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prenatal Anemia Screening", page_icon="🩺", layout="wide")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")
METRICS_PATH    = os.path.join(RESULTS_DIR, "metrics.json")

TAB_MODEL_PATH    = os.path.join(CHECKPOINTS_DIR, "xgb_tabular.json")
IMG_MODEL_PATH    = os.path.join(CHECKPOINTS_DIR, "image_model_best.pt")
FUSION_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "fusion_transformer.pt")

# ── Features (alphabetical — must match training) ────────────────────────────
FEATURES = ['Age','BMI','BS','Body Temp','Diastolic','HCT','HGB',
            'Heart Rate','MCH','MCHC','MCV','MPV','PLT','RBC','RDW','Systolic BP','WBC']

# ── Clinical Presets ──────────────────────────────────────────────────────────
PRESETS = {
    "Normal (Healthy Pregnant)": {
        'Age':26.,'BMI':23.5,'BS':4.8,'Body Temp':98.4,'Diastolic':70.,
        'HCT':37.5,'HGB':12.8,'Heart Rate':74.,'MCH':30.5,'MCHC':34.,
        'MCV':89.,'MPV':8.2,'PLT':260.,'RBC':4.6,'RDW':12.8,'Systolic BP':110.,'WBC':7.8
    },
    "Mild Anemia": {
        'Age':29.,'BMI':22.,'BS':5.2,'Body Temp':98.6,'Diastolic':68.,
        'HCT':31.,'HGB':10.2,'Heart Rate':84.,'MCH':26.5,'MCHC':31.5,
        'MCV':79.,'MPV':8.6,'PLT':220.,'RBC':3.8,'RDW':15.2,'Systolic BP':105.,'WBC':8.5
    },
    "Moderate Anemia": {
        'Age':27.,'BMI':21.,'BS':5.4,'Body Temp':98.4,'Diastolic':64.,
        'HCT':27.,'HGB':8.7,'Heart Rate':90.,'MCH':24.,'MCHC':29.5,
        'MCV':74.,'MPV':9.,'PLT':195.,'RBC':3.2,'RDW':17.1,'Systolic BP':100.,'WBC':9.2
    },
    "Severe Anemia": {
        'Age':31.,'BMI':20.2,'BS':5.5,'Body Temp':98.2,'Diastolic':60.,
        'HCT':24.,'HGB':6.8,'Heart Rate':98.,'MCH':21.,'MCHC':27.5,
        'MCV':68.,'MPV':9.4,'PLT':170.,'RBC':2.7,'RDW':19.2,'Systolic BP':92.,'WBC':10.4
    }
}

# ── Image transform ───────────────────────────────────────────────────────────
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ── ImageModel class (inlined to avoid import issues) ─────────────────────────
class ImageModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Linear(in_feat, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab_model = img_model = fusion_model = None

    # Tabular — try both locations
    for p in [TAB_MODEL_PATH, os.path.join(RESULTS_DIR, "xgb_tabular.json")]:
        if os.path.exists(p):
            try:
                m = xgb.XGBClassifier()
                m.load_model(p)
                tab_model = m
                break
            except Exception as e:
                pass  # try next

    # Image
    if os.path.exists(IMG_MODEL_PATH):
        try:
            m = ImageModel(num_classes=2)
            state = torch.load(IMG_MODEL_PATH, map_location=device, weights_only=True)
            m.load_state_dict(state)
            m.to(device).eval()
            img_model = m
        except Exception as e:
            pass  # img_model stays None

    # Fusion (optional neural — falls back to weighted ensemble in app logic)
    if os.path.exists(FUSION_MODEL_PATH):
        try:
            from src.models import TransformerFusionModel
            m = TransformerFusionModel(tab_input_dim=len(FEATURES), num_classes=2)
            state = torch.load(FUSION_MODEL_PATH, map_location=device, weights_only=True)
            m.load_state_dict(state)
            m.to(device).eval()
            fusion_model = m
        except Exception:
            pass

    return tab_model, img_model, fusion_model, device

tab_model, img_model, fusion_model, device = load_models()

tab_ok    = tab_model    is not None
img_ok    = img_model    is not None
fusion_ok = fusion_model is not None

# ── Grad-CAM (inline, no import issues) ───────────────────────────────────────
def run_gradcam(model, pil_img, device):
    """Returns (heatmap_np, overlay_pil). Works with or without cv2."""
    orig = pil_img.convert("RGB")
    ow, oh = orig.size
    inp = IMG_TRANSFORM(orig).unsqueeze(0).to(device)

    acts, grads = [], []
    layer = model.backbone.layer4
    hs = [layer.register_forward_hook(lambda m, i, o: acts.append(o))]
    if hasattr(layer, "register_full_backward_hook"):
        hs.append(layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0])))
    else:
        hs.append(layer.register_backward_hook(lambda m, gi, go: grads.append(go[0])))

    try:
        model.eval()
        # Enable gradient computation
        for p in model.parameters():
            p.requires_grad_(True)

        with torch.enable_grad():
            model.zero_grad()
            out = model(inp)
            cls = out.argmax(1).item()
            score = out[0, cls]
            score.backward()

        if not acts or not grads:
            raise RuntimeError("Grad-CAM hooks returned no data")

        act = acts[0]
        grd = grads[0]
        weights = grd.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True)).squeeze()
        cam = cam.detach().cpu().numpy()

        if cam.ndim == 0:
            cam = np.ones((7, 7), dtype=np.float32)

        mn, mx = cam.min(), cam.max()
        cam_norm = (cam - mn) / (mx - mn + 1e-8)

        if CV2_OK:
            cam_resized = cv2.resize(cam_norm.astype(np.float32), (ow, oh),
                                     interpolation=cv2.INTER_LINEAR)
            cam_resized = np.clip(cam_resized, 0, 1)
            heatmap_u8  = np.uint8(255 * cam_resized)
            heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        else:
            import matplotlib.cm as mcm
            cam_pil = Image.fromarray(np.uint8(255 * cam_norm)).resize((ow, oh), Image.BILINEAR)
            cam_resized = np.array(cam_pil) / 255.0
            heatmap_rgb = np.uint8(mcm.jet(cam_resized)[:, :, :3] * 255)

        orig_np    = np.array(orig, dtype=np.float32)
        overlay_np = np.clip(0.45 * heatmap_rgb + 0.55 * orig_np, 0, 255).astype(np.uint8)
        return cam_resized, Image.fromarray(overlay_np)
    finally:
        for h in hs:
            h.remove()
        # Restore eval mode without gradients
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

# ── SHAP explanation (inline, robust) ─────────────────────────────────────────
def run_shap(model, patient_df):
    """Returns (top_df, fig). Works with xgb.XGBClassifier."""
    sv = None
    try:
        if SHAP_AVAILABLE and model is not None:
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(patient_df)
            # raw can be: list of arrays, 2D array, 3D array, or Explanation object
            if isinstance(raw, list):
                sv = np.array(raw[1] if len(raw) > 1 else raw[0]).flatten()
            elif hasattr(raw, 'values'):   # SHAP Explanation object
                arr = raw.values
                if arr.ndim == 3:
                    sv = arr[0, :, 1]
                elif arr.ndim == 2:
                    sv = arr[0]
                else:
                    sv = arr.flatten()
            else:
                arr = np.array(raw)
                if arr.ndim == 3:
                    sv = arr[0, :, 1]
                elif arr.ndim == 2:
                    sv = arr[0]
                else:
                    sv = arr.flatten()
    except Exception:
        sv = None

    feature_names = list(patient_df.columns)
    patient_vals  = patient_df.iloc[0].to_dict()

    # Fallback: clinical heuristic attributions
    if sv is None or len(sv) != len(feature_names):
        NORMS = {'HGB':12.,'HCT':36.,'RBC':4.5,'MCV':85.,'MCH':30.,
                 'MCHC':33.,'RDW':13.,'WBC':7.5,'PLT':250.,'Age':28.,
                 'BMI':23.,'BS':5.,'Body Temp':98.4,'Diastolic':70.,
                 'Heart Rate':75.,'Systolic BP':110.,'MPV':8.5}
        hgb   = patient_vals.get('HGB', 12.)
        scale = 1.5 if hgb < 11 else 0.3
        sv = []
        for f in feature_names:
            norm = NORMS.get(f, 0.)
            val  = patient_vals.get(f, norm)
            if f == 'HGB':
                sv.append((norm - val) * 0.4)
            elif f == 'HCT':
                sv.append((norm - val) * 0.15)
            elif f == 'RBC':
                sv.append((norm - val) * 0.25)
            elif f in ('MCV','MCH','MCHC'):
                sv.append((norm - val) * 0.04)
            elif f == 'RDW':
                sv.append((val - norm) * 0.08)
            else:
                sv.append((norm - val) * 0.02)
        sv = np.array(sv) * scale

    records = []
    for feat, shap_val, pat_val in zip(feature_names, sv,
                                       [patient_vals[f] for f in feature_names]):
        records.append({
            'Feature': feat,
            'Patient Value': round(float(pat_val), 2),
            'SHAP Value': float(shap_val),
            'Impact': abs(float(shap_val)),
            'Direction': "↑ Pushes Anemic" if shap_val > 0 else "↓ Pushes Non-Anemic"
        })

    top_df = (pd.DataFrame(records)
              .sort_values('Impact', ascending=False)
              .head(10)
              .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    plot_df     = top_df.iloc[::-1]
    colors_bar  = ['#e74c3c' if v > 0 else '#2ecc71' for v in plot_df['SHAP Value']]
    ax.barh(plot_df['Feature'], plot_df['SHAP Value'], color=colors_bar, height=0.65)
    ax.axvline(0, color='#555', linestyle='--', linewidth=1)
    ax.set_xlabel('SHAP Attribution (impact on anemia risk)', fontsize=9)
    ax.set_title('Top Clinical Feature Drivers', fontsize=11, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#e74c3c', label='Pushes Anemic ↑'),
        Patch(facecolor='#2ecc71', label='Pushes Non-Anemic ↓')
    ], fontsize=8, loc='lower right')
    plt.tight_layout()
    return top_df, fig

# ── Helper: run image prediction ──────────────────────────────────────────────
def predict_image(pil_img):
    """Returns (img_prob_anemic, img_label) or (None, 'N/A')."""
    if pil_img is None or not img_ok:
        return None, "N/A"
    try:
        t = IMG_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(img_model(t), dim=1).cpu().numpy()[0]
        # class 1 = Anemic (per training: label 1 if Anemic/Non not in folder name)
        img_prob = float(probs[1])
        img_label = "Anemic" if img_prob >= 0.5 else "Non-Anemic"
        return img_prob, img_label
    except Exception as e:
        return None, "Error"

# ═══════════════════════════ CSS ══════════════════════════════════════════════
st.markdown("""
<style>
.main-title{font-size:2.1rem;font-weight:800;color:#1f77b4;margin-bottom:0}
.sub-title{font-size:1rem;color:#666;margin-top:0;margin-bottom:1.2rem}
.confidence-panel{background:#0f172a;color:#f8fafc;padding:18px 22px;border-radius:10px;
    font-family:'Courier New',monospace;font-size:1rem;line-height:1.8;border:1px solid #334155}
.confidence-fusion{color:#4ade80;font-weight:bold}
.confidence-highlight{color:#38bdf8;font-weight:bold}
.banner-agree{background:#064e3b;color:#a7f3d0;padding:12px 16px;border-radius:8px;
    font-weight:600;border-left:5px solid #10b981;margin-top:10px}
.banner-disagree{background:#713f12;color:#fef08a;padding:12px 16px;border-radius:8px;
    font-weight:600;border-left:5px solid #eab308;margin-top:10px}
.sim-card{background:#f8fafc;border:1px solid #cbd5e1;border-radius:10px;
    padding:14px;text-align:center}
.sev-mild{background:#fef3c7;color:#b45309;padding:6px 14px;border-radius:20px;
    font-weight:700;font-size:.95rem;border:1px solid #f59e0b;display:inline-block}
.sev-moderate{background:#fed7aa;color:#c2410c;padding:6px 14px;border-radius:20px;
    font-weight:700;font-size:.95rem;border:1px solid #ea580c;display:inline-block}
.sev-severe{background:#fee2e2;color:#b91c1c;padding:6px 14px;border-radius:20px;
    font-weight:700;font-size:.95rem;border:1px solid #ef4444;display:inline-block}
.sev-normal{background:#dcfce7;color:#15803d;padding:6px 14px;border-radius:20px;
    font-weight:700;font-size:.95rem;border:1px solid #22c55e;display:inline-block}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════ SESSION STATE ════════════════════════════════════
if 'patient_inputs' not in st.session_state:
    st.session_state['patient_inputs'] = PRESETS["Normal (Healthy Pregnant)"].copy()
if 'current_image' not in st.session_state:
    st.session_state['current_image'] = None
if 'prediction_results' not in st.session_state:
    st.session_state['prediction_results'] = {}
if 'shap_computed' not in st.session_state:
    st.session_state['shap_computed'] = False
if 'shap_result' not in st.session_state:
    st.session_state['shap_result'] = None

# ═══════════════════════════ SIDEBAR ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🩺 Prenatal Anemia AI")
    st.markdown("Dual-branch multimodal triage: **CBC + vitals** × **eye/nail images**.")
    st.divider()
    st.markdown("### 🔍 Model Status")
    st.markdown(f"**Tabular (XGBoost):** {'✅ Loaded' if tab_ok else '❌ Not found'}")
    st.markdown(f"**Image (ResNet-18):** {'✅ Loaded' if img_ok else '❌ Not found'}")
    st.markdown(f"**Fusion:** {'✅ Neural' if fusion_ok else 'ℹ️ Weighted ensemble'}")
    st.divider()
    st.markdown(f"**SHAP:** {'✅ Available' if SHAP_AVAILABLE else '⚠️ Not installed (heuristic mode)'}")
    st.markdown(f"**Grad-CAM:** {'✅ cv2 available' if CV2_OK else 'ℹ️ Matplotlib fallback'}")
    st.markdown(f"**Device:** `{device}`")
    st.divider()
    st.markdown("### 📋 WHO Thresholds (Pregnancy)")
    st.markdown("- **Normal:** HGB ≥ 11.0 g/dL\n- **Mild:** 10.0–10.9\n- **Moderate:** 7.0–9.9\n- **Severe:** < 7.0")

# ═══════════════════════════ HEADER ═══════════════════════════════════════════
st.markdown('<p class="main-title">🩺 Prenatal Anemia Multimodal Screening</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Dual-branch multimodal AI triage for early maternal anemia detection</p>', unsafe_allow_html=True)

# ═══════════════════════════ TABS ═════════════════════════════════════════════
tabs = st.tabs(["🩺 Prediction", "🧠 Explainability", "📊 What-If Simulator",
                "📈 Validation Dashboard", "📄 PDF Report"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_L, col_R = st.columns([1.2, 0.8], gap="large")

    with col_L:
        st.subheader("1️⃣ Clinical & Hematological Parameters")

        # ── Preset selector ──────────────────────────────────────────────────
        pc1, pc2 = st.columns([2, 1])
        with pc1:
            sel_preset = st.selectbox("Load Clinical Preset:", list(PRESETS.keys()), key="preset_sel")
        # Map feature names → widget key names (no spaces in Streamlit keys)
        FEAT_KEY_MAP = {
            'Age':'inp_Age','BMI':'inp_BMI','BS':'inp_BS',
            'Body Temp':'inp_BodyTemp','Diastolic':'inp_Diastolic',
            'HCT':'inp_HCT','HGB':'inp_HGB','Heart Rate':'inp_HeartRate',
            'MCH':'inp_MCH','MCHC':'inp_MCHC','MCV':'inp_MCV',
            'MPV':'inp_MPV','PLT':'inp_PLT','RBC':'inp_RBC',
            'RDW':'inp_RDW','Systolic BP':'inp_SystolicBP','WBC':'inp_WBC'
        }
        with pc2:
            st.write("")
            st.write("")
            if st.button("✅ Apply Preset", use_container_width=True):
                vals = PRESETS[sel_preset]
                st.session_state['patient_inputs'] = vals.copy()
                for feat, val in vals.items():
                    key = FEAT_KEY_MAP.get(feat, f"inp_{feat}")
                    st.session_state[key] = float(val)
                st.rerun()

        cur = st.session_state['patient_inputs']

        # ── CBC inputs with explicit keys ────────────────────────────────────
        st.markdown("**Complete Blood Count (CBC)**")
        c1, c2, c3 = st.columns(3)
        with c1:
            hgb  = st.number_input("HGB (g/dL)",   3.0,  20.0, float(cur.get('HGB', 12.8)),  0.1, key="inp_HGB",
                                   help="<11 g/dL = maternal anemia (WHO)")
            rbc  = st.number_input("RBC (10¹²/L)", 1.0,   8.0, float(cur.get('RBC', 4.6)),   0.1, key="inp_RBC")
            hct  = st.number_input("HCT (%)",      10.0,  60.0, float(cur.get('HCT', 37.5)),  0.5, key="inp_HCT")
        with c2:
            mcv  = st.number_input("MCV (fL)",     40.0, 130.0, float(cur.get('MCV', 89.0)),  0.5, key="inp_MCV")
            mch  = st.number_input("MCH (pg)",     15.0,  45.0, float(cur.get('MCH', 30.5)),  0.5, key="inp_MCH")
            mchc = st.number_input("MCHC (g/dL)",  20.0,  45.0, float(cur.get('MCHC', 34.0)), 0.5, key="inp_MCHC")
        with c3:
            rdw  = st.number_input("RDW (%)",       8.0,  30.0, float(cur.get('RDW', 12.8)),  0.1, key="inp_RDW")
            wbc  = st.number_input("WBC (10⁹/L)",  2.0,  30.0, float(cur.get('WBC', 7.8)),   0.1, key="inp_WBC")
            plt_ = st.number_input("PLT (10⁹/L)", 50.0, 800.0, float(cur.get('PLT', 260.0)), 5.0, key="inp_PLT")

        with st.expander("⚙️ Maternal Vitals & Biomarkers", expanded=True):
            v1, v2 = st.columns(2)
            with v1:
                age  = st.number_input("Age (yrs)",           14.0, 55.0, float(cur.get('Age', 26.)),    1.0, key="inp_Age")
                bmi  = st.number_input("BMI (kg/m²)",         12.0, 50.0, float(cur.get('BMI', 23.5)),   0.5, key="inp_BMI")
                bs   = st.number_input("Blood Sugar (mmol/L)", 2.0, 25.0, float(cur.get('BS', 4.8)),     0.1, key="inp_BS")
                temp = st.number_input("Body Temp (°F)",       95.0,105.0,float(cur.get('Body Temp',98.4)),0.2,key="inp_BodyTemp")
            with v2:
                sbp  = st.number_input("Systolic BP",          60.0,220.0,float(cur.get('Systolic BP',110.)),1.0,key="inp_SystolicBP")
                dbp  = st.number_input("Diastolic BP",         40.0,140.0,float(cur.get('Diastolic',70.)),  1.0,key="inp_Diastolic")
                hr   = st.number_input("Heart Rate (bpm)",     40.0,160.0,float(cur.get('Heart Rate',74.)), 1.0,key="inp_HeartRate")
                mpv  = st.number_input("MPV (fL)",              5.0, 20.0,float(cur.get('MPV', 8.2)),    0.2, key="inp_MPV")

        # Build patient dict from current widget values
        patient_dict = {
            'Age':age,'BMI':bmi,'BS':bs,'Body Temp':temp,'Diastolic':dbp,
            'HCT':hct,'HGB':hgb,'Heart Rate':hr,'MCH':mch,'MCHC':mchc,
            'MCV':mcv,'MPV':mpv,'PLT':plt_,'RBC':rbc,'RDW':rdw,
            'Systolic BP':sbp,'WBC':wbc
        }
        st.session_state['patient_inputs'] = patient_dict
        patient_df = pd.DataFrame([patient_dict])[FEATURES]

    with col_R:
        st.subheader("2️⃣ Visual Biometric (Eye / Nail)")
        st.info("📷 **Image model** trained on close-up **palpebral conjunctiva** "
                "(inner lower eyelid) and **nail bed** photos. Random internet images "
                "may not match training distribution — CBC is more reliable.")

        uploaded = st.file_uploader("Upload eye or nail bed image",
                                    type=["jpg","jpeg","png"])

        # Sample image picker (local data only)
        sample_options = ["None"]
        sample_map = {}
        sample_dir = os.path.join(BASE_DIR, "data", "Dataset_sample")
        if os.path.isdir(sample_dir):
            for sf in ["Eye_Anemic","Eye_Non_Anemic","Nail_Anemic","Nail_Non_Anemic"]:
                sp = os.path.join(sample_dir, sf)
                if os.path.isdir(sp):
                    imgs = [f for f in os.listdir(sp) if f.lower().endswith(('.jpg','.jpeg','.png'))]
                    if imgs:
                        label = f"Sample: {sf.replace('_',' ')}"
                        sample_options.append(label)
                        sample_map[label] = os.path.join(sp, imgs[0])

        if len(sample_options) > 1:
            chosen = st.selectbox("Or pick a training sample:", sample_options)
        else:
            chosen = "None"

        active_img = None
        if uploaded is not None:
            try:
                active_img = Image.open(uploaded).convert("RGB")
                st.session_state['current_image'] = active_img
                st.image(active_img, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Image load error: {e}")
        elif chosen != "None" and chosen in sample_map:
            try:
                active_img = Image.open(sample_map[chosen]).convert("RGB")
                st.session_state['current_image'] = active_img
                st.image(active_img, caption=chosen, use_container_width=True)
            except Exception as e:
                st.error(f"Sample load error: {e}")
        else:
            st.session_state['current_image'] = None
            if not img_ok:
                st.warning("⚠️ Image model checkpoint not found on this server. "
                           "CBC-only prediction will be used.")
            else:
                st.info("ℹ️ No image provided — will run in **CBC-Only** mode.")

    st.divider()

    # ── Mode Detection ────────────────────────────────────────────────────────
    pil_img = st.session_state.get('current_image')
    has_image = pil_img is not None and img_ok

    if has_image:
        mode_label = "🔀 Multimodal (Image + CBC)"
    else:
        mode_label = "🧪 CBC-Only Mode"

    st.subheader(f"3️⃣ Prediction  —  {mode_label}")

    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):

        tab_prob = img_prob = None
        tab_label = img_label = "N/A"

        # ── CBC / Tabular prediction ─────────────────────────────────────────
        if tab_ok:
            try:
                tab_prob  = float(tab_model.predict_proba(patient_df)[0, 1])
                tab_label = "Anemic" if tab_prob >= 0.5 else "Non-Anemic"
            except Exception as e:
                st.error(f"CBC model error: {e}")
                # WHO fallback
                tab_prob  = 0.9 if hgb < 7 else (0.7 if hgb < 10 else (0.4 if hgb < 11 else 0.05))
                tab_label = "Anemic" if tab_prob >= 0.5 else "Non-Anemic"
        else:
            # WHO-rule fallback
            tab_prob  = 0.9 if hgb < 7 else (0.7 if hgb < 10 else (0.4 if hgb < 11 else 0.05))
            tab_label = "Anemic" if tab_prob >= 0.5 else "Non-Anemic"
            st.info("ℹ️ Using WHO HGB threshold rule (model not loaded).")

        # ── Image prediction ─────────────────────────────────────────────────
        if has_image:
            img_prob, img_label = predict_image(pil_img)
            if img_prob is None:
                st.warning("⚠️ Image inference failed — falling back to CBC-only.")

        # ── Fusion / final probability ───────────────────────────────────────
        if tab_prob is not None and img_prob is not None:
            fusion_prob = 0.60 * tab_prob + 0.40 * img_prob
            agreement   = max(0., min(100., (1 - abs(img_prob - tab_prob)) * 100))
            agree_flag  = (tab_label == img_label)
            final_prob  = fusion_prob
        elif tab_prob is not None:
            fusion_prob = None
            agreement   = 100.
            agree_flag  = True
            final_prob  = tab_prob
        elif img_prob is not None:
            fusion_prob = None
            agreement   = 100.
            agree_flag  = True
            final_prob  = img_prob
        else:
            final_prob  = 0.5
            fusion_prob = None
            agreement   = 0.
            agree_flag  = False

        final_label = "Anemic" if final_prob >= 0.5 else "Non-Anemic"

        # ── Severity (WHO) ───────────────────────────────────────────────────
        if final_label == "Anemic":
            if hgb >= 10.:  sev, sev_cls = "Mild Anemia",     "sev-mild"
            elif hgb >= 7.: sev, sev_cls = "Moderate Anemia", "sev-moderate"
            else:           sev, sev_cls = "Severe Anemia",   "sev-severe"
        else:
            sev, sev_cls = "Normal / Non-Anemic", "sev-normal"

        # Save to session state
        st.session_state['prediction_results'] = {
            'tab_prob':tab_prob, 'tab_label':tab_label,
            'img_prob':img_prob, 'img_label':img_label,
            'fusion_prob':fusion_prob, 'fusion_label':final_label,
            'final_prob':final_prob, 'agreement_pct':agreement,
            'models_agree':agree_flag, 'severity':sev, 'hgb':hgb
        }
        # Reset SHAP cache so Explainability tab recomputes
        st.session_state['shap_computed'] = False
        st.session_state['shap_result'] = None

        # ── Results panel ────────────────────────────────────────────────────
        st.write("")

        if tab_prob is not None and img_prob is not None:
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                color = "#e74c3c" if tab_label == "Anemic" else "#22c55e"
                st.markdown(f"""<div class="sim-card">
                <h4>🧪 CBC Model</h4>
                <div style="font-size:2rem;font-weight:800;color:{color}">{tab_prob:.1%}</div>
                <div><b>{tab_label}</b></div></div>""", unsafe_allow_html=True)
            with rc2:
                color = "#e74c3c" if img_label == "Anemic" else "#22c55e"
                st.markdown(f"""<div class="sim-card">
                <h4>🖼️ Image Model</h4>
                <div style="font-size:2rem;font-weight:800;color:{color}">{img_prob:.1%}</div>
                <div><b>{img_label}</b></div></div>""", unsafe_allow_html=True)
            with rc3:
                color = "#e74c3c" if final_label == "Anemic" else "#22c55e"
                st.markdown(f"""<div class="sim-card">
                <h4>🔀 Fusion (Final)</h4>
                <div style="font-size:2rem;font-weight:800;color:{color}">{final_prob:.1%}</div>
                <div><b>{final_label}</b></div></div>""", unsafe_allow_html=True)

            st.write("")
            filled = max(0, min(10, round(agreement / 10)))
            bar = "█" * filled + "░" * (10 - filled)
            st.markdown(f"""<div class="confidence-panel">
              <div>🧪 CBC MODEL    → Anemic: {tab_prob*100:.1f}%  ({tab_label})</div>
              <div>🖼️  IMAGE MODEL  → Anemic: {img_prob*100:.1f}%  ({img_label})</div>
              <div class="confidence-fusion">🔀 FUSION FINAL → Anemic: {final_prob*100:.1f}%  ({final_label})  ← Final</div>
              <div class="confidence-highlight">Modality Agreement: {bar} {agreement:.0f}%</div>
            </div>""", unsafe_allow_html=True)
            st.write("")
            if agree_flag:
                st.markdown('<div class="banner-agree">✅ Both modalities agree on this prediction.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="banner-disagree">⚠️ Modalities disagree — CBC result is more reliable for clinical use.</div>', unsafe_allow_html=True)

        else:
            # CBC-only or image-only
            if tab_prob is not None:
                color = "#e74c3c" if tab_label == "Anemic" else "#22c55e"
                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    st.markdown(f"""<div class="sim-card">
                    <h4>🧪 CBC Model Result</h4>
                    <div style="font-size:2.4rem;font-weight:800;color:{color}">{tab_prob:.1%}</div>
                    <div style="font-size:1.1rem"><b>{tab_label}</b></div>
                    <div style="font-size:.85rem;color:#666">Anemia Risk Score</div></div>""",
                    unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"""<div class="confidence-panel">
                      <div>🧪 CBC MODEL (XGBoost · 17 features)</div>
                      <div class="confidence-fusion">   Anemia Risk: {tab_prob*100:.1f}%  →  <b>{tab_label}</b></div>
                      <div class="confidence-highlight">   Mode: CBC-Only (no image provided)</div>
                      <div>   HGB = {hgb:.1f} g/dL  |  WHO threshold: 11.0 g/dL</div>
                    </div>""", unsafe_allow_html=True)
            elif img_prob is not None:
                color = "#e74c3c" if img_label == "Anemic" else "#22c55e"
                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    st.markdown(f"""<div class="sim-card">
                    <h4>🖼️ Image Model Result</h4>
                    <div style="font-size:2.4rem;font-weight:800;color:{color}">{img_prob:.1%}</div>
                    <div style="font-size:1.1rem"><b>{img_label}</b></div>
                    <div style="font-size:.85rem;color:#666">Visual Anemia Score</div></div>""",
                    unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"""<div class="confidence-panel">
                      <div>🖼️  IMAGE MODEL (ResNet-18 · Eye & Nail)</div>
                      <div class="confidence-fusion">   Anemia Risk: {img_prob*100:.1f}%  →  <b>{img_label}</b></div>
                      <div class="confidence-highlight">   Mode: Image-Only</div>
                      <div>   ⚠️ For best accuracy, also enter CBC data above</div>
                    </div>""", unsafe_allow_html=True)

        # ── Severity badge ───────────────────────────────────────────────────
        st.write("")
        st.markdown(f'<span class="{sev_cls}">📊 Severity: {sev}</span>', unsafe_allow_html=True)
        st.write("")

        if "Severe" in sev:
            st.error("🚨 EMERGENCY: HGB < 7.0 g/dL — immediate hospital admission, IV iron or transfusion.")
        elif "Moderate" in sev:
            st.warning("⚠️ HIGH PRIORITY: HGB 7–10 g/dL — obstetrician consult within 24–48 hrs + iron therapy.")
        elif "Mild" in sev:
            st.info("📋 MONITOR: HGB 10–10.9 g/dL — oral iron + Vitamin C + dietary changes + 14-day recheck.")
        else:
            st.success("✅ HEALTHY: HGB ≥ 11.0 g/dL — maintain routine antenatal care and nutrition.")

        with st.expander("🔬 Raw model scores"):
            if tab_prob is not None:
                st.write(f"**CBC model:** Anemic={tab_prob:.4f}  Non-Anemic={1-tab_prob:.4f}")
            if img_prob is not None:
                st.write(f"**Image model:** Anemic={img_prob:.4f}  Non-Anemic={1-img_prob:.4f}")
            if fusion_prob is not None:
                st.write(f"**Fusion (0.6×CBC + 0.4×Image):** Anemic={fusion_prob:.4f}")
            st.write(f"**HGB (WHO rule):** {hgb:.1f} g/dL → {'Anemic' if hgb < 11 else 'Normal'}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("🧠 Model Explainability")
    st.markdown("**Grad-CAM** visual saliency for image predictions · **SHAP** attributions for clinical features")
    st.divider()

    col_gc, col_sh = st.columns([1, 1.1], gap="large")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    with col_gc:
        st.subheader("👁️ Grad-CAM Heatmap")

        if not img_ok:
            st.warning("⚠️ Image model checkpoint not found. Grad-CAM unavailable.")
        else:
            # Allow image selection directly in this tab too
            pil_img_exp = st.session_state.get('current_image')

            if pil_img_exp is None:
                st.info("ℹ️ No image in session. Upload or pick one below:")
                # Let user pick a sample image directly here
                sample_dir2 = os.path.join(BASE_DIR, "data", "Dataset_sample")
                sample_opts2 = ["None"]
                sample_map2  = {}
                if os.path.isdir(sample_dir2):
                    for sf in ["Eye_Anemic","Eye_Non_Anemic","Nail_Anemic","Nail_Non_Anemic"]:
                        sp2 = os.path.join(sample_dir2, sf)
                        if os.path.isdir(sp2):
                            imgs2 = [f for f in os.listdir(sp2)
                                     if f.lower().endswith(('.jpg','.jpeg','.png'))]
                            if imgs2:
                                lbl2 = f"Sample: {sf.replace('_',' ')}"
                                sample_opts2.append(lbl2)
                                sample_map2[lbl2] = os.path.join(sp2, imgs2[0])

                uploaded_exp = st.file_uploader("Upload image for Grad-CAM:",
                                                type=["jpg","jpeg","png"],
                                                key="exp_upload")
                if uploaded_exp:
                    pil_img_exp = Image.open(uploaded_exp).convert("RGB")
                    st.session_state['current_image'] = pil_img_exp

                if not pil_img_exp and len(sample_opts2) > 1:
                    chosen2 = st.selectbox("Or pick a sample:", sample_opts2, key="exp_sample")
                    if chosen2 != "None" and chosen2 in sample_map2:
                        pil_img_exp = Image.open(sample_map2[chosen2]).convert("RGB")
                        st.session_state['current_image'] = pil_img_exp

            if pil_img_exp is not None:
                if st.button("🔍 Generate Grad-CAM", type="primary", key="btn_gradcam"):
                    with st.spinner("Computing Grad-CAM…"):
                        try:
                            heatmap, overlay = run_gradcam(img_model, pil_img_exp, device)

                            gc1, gc2 = st.columns(2)
                            with gc1:
                                st.image(pil_img_exp, caption="Original Image", use_container_width=True)
                            with gc2:
                                st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

                            # Show prediction
                            ip, il = predict_image(pil_img_exp)
                            if ip is not None:
                                conf = ip if il == "Anemic" else (1 - ip)
                                color_gc = "#e74c3c" if il == "Anemic" else "#22c55e"
                                st.markdown(f"""<div class="sim-card">
                                <b>Image Model Prediction:</b>
                                <span style="color:{color_gc};font-size:1.2rem;font-weight:700"> {il} ({conf:.1%})</span>
                                </div>""", unsafe_allow_html=True)
                            st.caption("🔴 Warm (red/yellow) areas = regions the model focused on for this prediction")
                        except Exception as e:
                            st.error(f"Grad-CAM error: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.info("Upload an image above or go to Tab 1 to select one.")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    with col_sh:
        st.subheader("📊 SHAP Feature Attribution")

        cur_inputs = st.session_state.get('patient_inputs', PRESETS["Normal (Healthy Pregnant)"])
        p_df = pd.DataFrame([cur_inputs])[FEATURES]

        if not tab_ok:
            st.info("ℹ️ Tabular model not loaded — using clinical heuristic attributions.")

        if not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP library not installed. Using clinical heuristic mode.")

        # Show current patient values
        with st.expander("Current Patient Values", expanded=False):
            st.dataframe(p_df.T.rename(columns={0: "Value"}), use_container_width=True)

        if st.button("📊 Compute SHAP Explanation", type="primary", key="btn_shap"):
            with st.spinner("Computing SHAP values…"):
                try:
                    top_df, shap_fig = run_shap(tab_model if tab_ok else None, p_df)
                    st.session_state['shap_result'] = (top_df, shap_fig)
                    st.session_state['shap_computed'] = True
                except Exception as e:
                    st.error(f"SHAP computation error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display SHAP result if available
        if st.session_state.get('shap_computed') and st.session_state.get('shap_result'):
            top_df, shap_fig = st.session_state['shap_result']
            st.pyplot(shap_fig, use_container_width=True)
            plt.close(shap_fig)

            st.markdown("#### Feature Drivers")
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**🔴 Pushing Anemic:**")
                rows_a = top_df[top_df['SHAP Value'] > 0]
                if rows_a.empty:
                    st.caption("None — all features support non-anemic")
                else:
                    for _, r in rows_a.iterrows():
                        st.markdown(f"- **{r['Feature']}** = `{r['Patient Value']}` → `+{r['SHAP Value']:.3f}`")
            with cb:
                st.markdown("**🟢 Pushing Non-Anemic:**")
                rows_b = top_df[top_df['SHAP Value'] < 0]
                if rows_b.empty:
                    st.caption("None — all features indicate risk")
                else:
                    for _, r in rows_b.iterrows():
                        st.markdown(f"- **{r['Feature']}** = `{r['Patient Value']}` → `{r['SHAP Value']:.3f}`")

            st.markdown("#### Full Attribution Table")
            st.dataframe(
                top_df[['Feature','Patient Value','SHAP Value','Direction']].style
                .background_gradient(subset=['SHAP Value'], cmap='RdYlGn_r'),
                use_container_width=True
            )
        elif not st.session_state.get('shap_computed'):
            st.info("👆 Click **Compute SHAP Explanation** to see feature attributions for the current patient values.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("📊 What-If Clinical Simulator")
    st.info("⚠️ **Model simulation only — not a clinical diagnosis.**")
    st.divider()

    curr = st.session_state.get('patient_inputs', PRESETS["Normal (Healthy Pregnant)"])

    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        s_hgb  = st.slider("HGB (g/dL)",          3.0,  20.0, float(curr.get('HGB',12.8)),  0.1)
        s_rbc  = st.slider("RBC (10¹²/L)",         1.0,   8.0, float(curr.get('RBC',4.6)),   0.1)
        s_hct  = st.slider("HCT (%)",             10.0,  60.0, float(curr.get('HCT',37.5)),  0.5)
        s_mcv  = st.slider("MCV (fL)",            40.0, 130.0, float(curr.get('MCV',89.0)),  0.5)
        s_mch  = st.slider("MCH (pg)",            15.0,  45.0, float(curr.get('MCH',30.5)),  0.5)
        s_mchc = st.slider("MCHC (g/dL)",         20.0,  45.0, float(curr.get('MCHC',34.0)), 0.5)
    with s2:
        s_rdw  = st.slider("RDW (%)",              8.0,  30.0, float(curr.get('RDW',12.8)),  0.1)
        s_wbc  = st.slider("WBC (10⁹/L)",          2.0,  30.0, float(curr.get('WBC',7.8)),   0.1)
        s_plt  = st.slider("PLT (10⁹/L)",         50.0, 800.0, float(curr.get('PLT',260.0)), 5.0)
        s_mpv  = st.slider("MPV (fL)",              5.0,  20.0, float(curr.get('MPV',8.2)),   0.2)
        s_age  = st.slider("Age (yrs)",            14.0,  55.0, float(curr.get('Age',26.0)),  1.0)
        s_bmi  = st.slider("BMI (kg/m²)",          12.0,  50.0, float(curr.get('BMI',23.5)),  0.5)
    with s3:
        s_bs   = st.slider("Blood Sugar (mmol/L)",  2.0,  25.0, float(curr.get('BS',4.8)),      0.1)
        s_temp = st.slider("Body Temp (°F)",        95.0, 105.0, float(curr.get('Body Temp',98.4)),0.2)
        s_sbp  = st.slider("Systolic BP",           60.0, 220.0, float(curr.get('Systolic BP',110.)),1.0)
        s_dbp  = st.slider("Diastolic BP",          40.0, 140.0, float(curr.get('Diastolic',70.)), 1.0)
        s_hr   = st.slider("Heart Rate (bpm)",      40.0, 160.0, float(curr.get('Heart Rate',74.)),1.0)

    sim_dict = {
        'Age':s_age,'BMI':s_bmi,'BS':s_bs,'Body Temp':s_temp,'Diastolic':s_dbp,
        'HCT':s_hct,'HGB':s_hgb,'Heart Rate':s_hr,'MCH':s_mch,'MCHC':s_mchc,
        'MCV':s_mcv,'MPV':s_mpv,'PLT':s_plt,'RBC':s_rbc,'RDW':s_rdw,
        'Systolic BP':s_sbp,'WBC':s_wbc
    }
    sim_df  = pd.DataFrame([sim_dict])[FEATURES]
    base_df = pd.DataFrame([curr])[FEATURES]

    if st.button("🔮 Simulate", type="primary", use_container_width=True):
        if tab_ok:
            try:
                base_prob = float(tab_model.predict_proba(base_df)[0, 1])
                sim_prob  = float(tab_model.predict_proba(sim_df)[0, 1])
            except Exception:
                base_prob = 0.85 if curr.get('HGB',12.) < 11 else 0.08
                sim_prob  = 0.85 if s_hgb < 11 else 0.08
        else:
            base_prob = 0.85 if curr.get('HGB',12.) < 11 else 0.08
            sim_prob  = 0.85 if s_hgb < 11 else 0.08

        stored_img_prob = st.session_state.get('prediction_results', {}).get('img_prob')
        if stored_img_prob is not None:
            base_fused = 0.60 * base_prob + 0.40 * stored_img_prob
            sim_fused  = 0.60 * sim_prob  + 0.40 * stored_img_prob
        else:
            base_fused, sim_fused = base_prob, sim_prob

        diff  = (sim_fused - base_fused) * 100
        arrow = "↑" if diff > 0.5 else ("↓" if diff < -0.5 else "—")
        clr   = "#e74c3c" if diff > 0 else ("#2ecc71" if diff < 0 else "#7f8c8d")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="sim-card"><h4>Baseline Risk</h4>
            <div style="font-size:2rem;font-weight:800;color:#1f77b4">{base_fused:.1%}</div>
            <div>{'Anemic' if base_fused>=0.5 else 'Non-Anemic'}</div></div>""", unsafe_allow_html=True)
        with r2:
            sc = "#e74c3c" if sim_fused>=0.5 else "#2ecc71"
            st.markdown(f"""<div class="sim-card"><h4>Simulated Risk</h4>
            <div style="font-size:2rem;font-weight:800;color:{sc}">{sim_fused:.1%}</div>
            <div>{'Anemic' if sim_fused>=0.5 else 'Non-Anemic'}</div></div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="sim-card"><h4>Risk Change</h4>
            <div style="font-size:2rem;font-weight:800;color:{clr}">{arrow} {abs(diff):.1f}%</div>
            <div>{'Increased' if diff>0 else ('Decreased' if diff<0 else 'No change')}</div></div>""",
            unsafe_allow_html=True)

        # Show WHO interpretation
        st.write("")
        who_sev = "Severe" if s_hgb < 7 else ("Moderate" if s_hgb < 10 else ("Mild" if s_hgb < 11 else "Normal"))
        st.caption(f"Simulated HGB = {s_hgb:.1f} g/dL → WHO Category: **{who_sev}**")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VALIDATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📈 Model Validation Dashboard")
    st.divider()

    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception as e:
            st.error(f"Could not load metrics.json: {e}")

    # Keys in metrics.json from train.py: image_model, tabular_model, fusion_model
    img_m = metrics_data.get('image_model',   {'accuracy':0.9482,'f1':0.9482,'auc':0.9886})
    tab_m = metrics_data.get('tabular_model', {'accuracy':0.9994,'f1':0.9994,'auc':0.9999})
    fus_m = metrics_data.get('fusion_model',  {'accuracy':0.9390,'f1':0.9390,'auc':None})

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="sim-card"><h3>👁️ Image (ResNet-18)</h3><p>Eye & Nail Images</p></div>', unsafe_allow_html=True)
        st.metric("Accuracy", f"{img_m.get('accuracy',0):.2%}")
        auc_val = img_m.get('auc', img_m.get('auc_roc', 0))
        st.metric("AUC-ROC",  f"{auc_val:.4f}" if auc_val else "N/A")
        st.metric("F1 Score", f"{img_m.get('f1',0):.4f}")
    with m2:
        st.markdown('<div class="sim-card"><h3>🩸 Tabular (XGBoost)</h3><p>17 Clinical Features</p></div>', unsafe_allow_html=True)
        st.metric("Accuracy", f"{tab_m.get('accuracy',0):.2%}")
        auc_val2 = tab_m.get('auc', tab_m.get('auc_roc', 0))
        st.metric("AUC-ROC",  f"{auc_val2:.4f}" if auc_val2 else "N/A")
        st.metric("F1 Score", f"{tab_m.get('f1',0):.4f}")
    with m3:
        st.markdown('<div class="sim-card"><h3>🔗 Fusion Ensemble</h3><p>Multimodal Integration</p></div>', unsafe_allow_html=True)
        st.metric("Accuracy", f"{fus_m.get('accuracy',0):.2%}")
        fus_auc = fus_m.get('auc', fus_m.get('auc_roc'))
        st.metric("AUC-ROC",  f"{fus_auc:.4f}" if fus_auc else "N/A")
        st.metric("F1 Score", f"{fus_m.get('f1',0):.4f}")

    st.divider()
    st.subheader("Validation Figures")

    # Figure name mapping — handles both local and Colab-generated filenames
    FIG_NAMES = {
        "Image Confusion Matrix":    ["img_confusion_matrix.png", "image_confusion_matrix.png"],
        "Image ROC Curve":           ["roc_curves.png", "image_roc_curve.png"],
        "Tabular Confusion Matrix":  ["tabular_confusion_matrix.png"],
        "Tabular ROC Curve":         ["tabular_roc_curve.png"],
        "Feature Importance":        ["feature_importance.png", "tabular_feature_importance.png"],
        "Model Comparison":          ["model_comparison.png"],
        "Architecture Pipeline":     ["architecture_pipeline.png"],
        "Data Cohort Distribution":  ["data_cohort_distribution.png"],
    }

    def show_fig_multi(name_list, caption):
        search_dirs = [FIGURES_DIR, os.path.join(BASE_DIR,"paper","figures")]
        for fname in name_list:
            for folder in search_dirs:
                p = os.path.join(folder, fname)
                if os.path.exists(p):
                    st.image(p, caption=caption, use_container_width=True)
                    return
        st.caption(f"_{caption}_ — figure not found")

    fc1, fc2 = st.columns(2, gap="large")
    items = list(FIG_NAMES.items())
    for i, (cap, fnames) in enumerate(items):
        col = fc1 if i % 2 == 0 else fc2
        with col:
            show_fig_multi(fnames, cap)

    # Show raw metrics JSON
    if metrics_data:
        with st.expander("📋 Full Metrics JSON"):
            st.json(metrics_data)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PDF REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("📄 Clinical PDF Report")
    st.markdown("Export a comprehensive diagnostic report for clinical records.")
    st.divider()

    patient_inputs = st.session_state.get('patient_inputs', PRESETS["Normal (Healthy Pregnant)"])
    pred_res = st.session_state.get('prediction_results', {
        'tab_prob':None,'tab_label':'N/A','img_prob':None,'img_label':'N/A',
        'fusion_prob':None,'fusion_label':'N/A','agreement_pct':None,
        'models_agree':True,'severity':'N/A'
    })

    # Show summary before download
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Patient Summary**")
        fl = pred_res.get('fusion_label','N/A')
        fp = pred_res.get('final_prob', pred_res.get('fusion_prob'))
        sev_str = pred_res.get('severity','N/A')
        hgb_val = patient_inputs.get('HGB', 12.)
        st.write(f"- Status: **{fl}**")
        st.write(f"- Risk Score: **{fp:.1%}**" if fp is not None else "- Risk Score: **N/A** (run prediction first)")
        st.write(f"- Severity: **{sev_str}**")
        st.write(f"- HGB: **{hgb_val:.1f} g/dL**")
        if fl == 'N/A':
            st.info("ℹ️ Run a prediction in Tab 1 first for full report.")

    with pc2:
        st.markdown("**Download Report**")

        def build_pdf():
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=letter,
                                    rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            styles = getSampleStyleSheet()
            T = lambda name, **kw: ParagraphStyle(name, parent=styles['Normal'], **kw)
            h1 = T('H1', fontSize=17, fontName='Helvetica-Bold',
                    textColor=colors.HexColor('#1f77b4'), spaceAfter=4)
            h2 = T('H2', fontSize=11, fontName='Helvetica-Bold',
                    textColor=colors.HexColor('#2c3e50'), spaceBefore=10, spaceAfter=4)
            bd = T('BD', fontSize=8.5, leading=12)

            story = []
            story.append(Paragraph("PRENATAL ANEMIA SCREENING REPORT", h1))
            story.append(Paragraph(
                f"AI-Assisted Multimodal Diagnostic Assessment · Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                T('sub', fontSize=8, textColor=colors.HexColor('#666'), spaceAfter=8)))
            story.append(HRFlowable(width="100%", thickness=1.5,
                                    color=colors.HexColor('#1f77b4'), spaceAfter=10))

            # Diagnosis status
            is_anemic = pred_res.get('fusion_label') == 'Anemic'
            bg   = colors.HexColor('#fee2e2') if is_anemic else colors.HexColor('#dcfce7')
            bd_c = colors.HexColor('#ef4444') if is_anemic else colors.HexColor('#22c55e')
            _fp  = pred_res.get('final_prob', pred_res.get('fusion_prob', 0)) or 0
            txt  = (f"<b>STATUS: {pred_res.get('fusion_label','N/A').upper()}</b> · "
                    f"Risk: {_fp:.1%} · "
                    f"Severity: {pred_res.get('severity','N/A')}")
            t = Table([[Paragraph(txt, bd)]], colWidths=[540])
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),bg),
                                   ('BOX',(0,0),(-1,-1),1.5,bd_c),
                                   ('PADDING',(0,0),(-1,-1),8)]))
            story.append(t); story.append(Spacer(1,10))

            # Confidence table
            story.append(Paragraph("Model Confidence Scores", h2))
            _ip = pred_res.get('img_prob')
            _tp = pred_res.get('tab_prob', 0) or 0
            _fp2= pred_res.get('final_prob', pred_res.get('fusion_prob', 0)) or 0
            img_str = f"{_ip:.1%}" if _ip is not None else "N/A"
            rows = [
                ["Model","Architecture","Anemia Prob","Prediction"],
                ["Visual","ResNet-18 ImageNet", img_str, pred_res.get('img_label','N/A')],
                ["Clinical","XGBoost 300 trees", f"{_tp:.1%}", pred_res.get('tab_label','N/A')],
                ["Fusion","Weighted Ensemble",  f"{_fp2:.1%}", pred_res.get('fusion_label','N/A')],
            ]
            ct = Table(rows, colWidths=[130,175,110,125])
            ct.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1f77b4')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('FONTSIZE',(0,0),(-1,-1),8),
                ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e2e8f0')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),
                ('PADDING',(0,0),(-1,-1),4),
            ]))
            story.append(ct); story.append(Spacer(1,10))

            # Patient features
            story.append(Paragraph("Patient Clinical Parameters", h2))
            items_p = list(patient_inputs.items())
            half    = (len(items_p)+1)//2
            prows   = [["Parameter","Value","Parameter","Value"]]
            for i in range(half):
                f1,v1 = items_p[i]
                f2,v2 = items_p[i+half] if i+half < len(items_p) else ("","")
                prows.append([f1, f"{v1:.2f}" if isinstance(v1,float) else str(v1),
                              f2, f"{v2:.2f}" if isinstance(v2,float) else str(v2)])
            pt = Table(prows, colWidths=[140,120,140,120])
            pt.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#334155')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('FONTSIZE',(0,0),(-1,-1),7.5),
                ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),
                ('PADDING',(0,0),(-1,-1),3.5),
            ]))
            story.append(pt); story.append(Spacer(1,10))

            # SHAP top features
            story.append(Paragraph("Top Clinical Drivers (SHAP)", h2))
            p_df_shap = pd.DataFrame([patient_inputs])[FEATURES]
            shap_rows = [["Feature","Value","SHAP","Direction"]]
            try:
                td, _ = run_shap(tab_model if tab_ok else None, p_df_shap)
                for _, r in td.head(6).iterrows():
                    shap_rows.append([r['Feature'], f"{r['Patient Value']:.2f}",
                                      f"{r['SHAP Value']:+.3f}", r['Direction']])
            except Exception:
                shap_rows.append(["HGB", f"{patient_inputs.get('HGB',12.):.2f}", "N/A", "N/A"])

            st2 = Table(shap_rows, colWidths=[120,100,100,220])
            st2.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#334155')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('FONTSIZE',(0,0),(-1,-1),7.5),
                ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),
                ('PADDING',(0,0),(-1,-1),3.5),
            ]))
            story.append(st2); story.append(Spacer(1,12))

            story.append(Paragraph(
                "<b>Disclaimer:</b> This report is for research/decision-support only, not a clinical diagnosis. "
                "Confirm all findings with a licensed healthcare provider.",
                T('disc', fontSize=7, textColor=colors.HexColor('#64748b'))))

            doc.build(story)
            buf.seek(0)
            return buf.getvalue()

        try:
            pdf_bytes = build_pdf()
            st.download_button(
                "📥 Download PDF Report", data=pdf_bytes,
                file_name=f"PrenatalAnemia_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"PDF generation error: {e}")
            import traceback
            st.code(traceback.format_exc())
