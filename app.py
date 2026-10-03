# IMPORTANT: torch must be imported FIRST on Windows to avoid DLL conflicts
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from PIL import Image
import json

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prenatal Anemia Screening",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Paths (relative — works both locally and on Streamlit Cloud) ───────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_MODEL_PATH = os.path.join(RESULTS_DIR, "xgb_tabular.json")
IMG_MODEL_PATH = os.path.join(RESULTS_DIR, "image_model_best.pt")
METRICS_PATH   = os.path.join(RESULTS_DIR, "metrics.json")

# ─── Feature list (must match training order) ────────────────────────────────
FEATURES = ['Age', 'BMI', 'BS', 'Body Temp', 'Diastolic', 'HCT', 'HGB',
            'Heart Rate', 'MCH', 'MCHC', 'MCV', 'MPV', 'PLT', 'RBC',
            'RDW', 'Systolic BP', 'WBC']

# ─── Clinical Presets ────────────────────────────────────────────────────────
PRESETS = {
    "Normal (Healthy Pregnant)": {
        'Age': 26.0, 'BMI': 23.5, 'BS': 4.8, 'Body Temp': 98.4,
        'Diastolic': 70.0, 'HCT': 37.5, 'HGB': 12.8, 'Heart Rate': 74.0,
        'MCH': 30.5, 'MCHC': 34.0, 'MCV': 89.0, 'MPV': 8.2,
        'PLT': 260.0, 'RBC': 4.6, 'RDW': 12.8, 'Systolic BP': 110.0, 'WBC': 7.8
    },
    "Mild Anemia": {
        'Age': 29.0, 'BMI': 22.0, 'BS': 5.2, 'Body Temp': 98.6,
        'Diastolic': 68.0, 'HCT': 31.0, 'HGB': 10.2, 'Heart Rate': 84.0,
        'MCH': 26.5, 'MCHC': 31.5, 'MCV': 79.0, 'MPV': 8.6,
        'PLT': 220.0, 'RBC': 3.8, 'RDW': 15.2, 'Systolic BP': 105.0, 'WBC': 8.5
    },
    "Severe Anemia": {
        'Age': 31.0, 'BMI': 20.2, 'BS': 5.5, 'Body Temp': 98.2,
        'Diastolic': 60.0, 'HCT': 24.0, 'HGB': 7.6, 'Heart Rate': 96.0,
        'MCH': 22.0, 'MCHC': 28.0, 'MCV': 70.0, 'MPV': 9.2,
        'PLT': 180.0, 'RBC': 2.9, 'RDW': 18.5, 'Systolic BP': 95.0, 'WBC': 9.8
    },
}

# ─── Image Transform ─────────────────────────────────────────────────────────
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── ImageModel class (mirrors src/models.py) ────────────────────────────────
class ImageModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ─── Load Models (cached so they load only once per session) ─────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    tab_model = img_model = device = None

    # Tabular
    if os.path.exists(TAB_MODEL_PATH):
        try:
            tab_model = xgb.XGBClassifier()
            tab_model.load_model(TAB_MODEL_PATH)
        except Exception as e:
            st.error(f"Tabular model error: {e}")

    # Image
    if os.path.exists(IMG_MODEL_PATH):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img_model = ImageModel(num_classes=2, pretrained=False)
            img_model.load_state_dict(
                torch.load(IMG_MODEL_PATH, map_location=device)
            )
            img_model.to(device).eval()
        except Exception as e:
            st.error(f"Image model error: {e}")

    return tab_model, img_model, device

tab_model, img_model, device = load_models()

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.big-title { font-size:2.4rem; font-weight:800; color:#A23B72; margin-bottom:0; }
.sub-title  { font-size:1.05rem; color:#888; margin-top:0; margin-bottom:1.5rem; }
.risk-high  { background:#5c0a1e; border-left:5px solid #e74c3c; padding:18px;
              border-radius:8px; font-size:1.15rem; font-weight:700; color:#ff6b6b; }
.risk-ok    { background:#0a3a1e; border-left:5px solid #2ecc71; padding:18px;
              border-radius:8px; font-size:1.15rem; font-weight:700; color:#2ecc71; }
.metric-box { background:#1E2130; border-radius:10px; padding:18px; text-align:center; }
.metric-val { font-size:2rem; font-weight:800; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<p class="big-title">🩸 Prenatal Anemia Multimodal Screening</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-powered early detection combining blood test data & conjunctival/nail images</p>', unsafe_allow_html=True)
st.divider()

# ─── Navigation Tabs ─────────────────────────────────────────────────────────
tab_screen, tab_guide, tab_paper = st.tabs([
    "🔬 Clinical Screening",
    "🥗 Health & Diet Guidelines",
    "📊 Model Performance"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCREENING
# ═════════════════════════════════════════════════════════════════════════════
with tab_screen:
    col_left, col_right = st.columns([1.2, 1.0], gap="large")

    # ── Clinical Inputs ───────────────────────────────────────────────────
    with col_left:
        st.subheader("1️⃣ Clinical & Hematological Parameters")

        preset = st.selectbox("Quick Demo Preset:", list(PRESETS.keys()))
        D = PRESETS[preset]

        c1, c2 = st.columns(2)
        with c1:
            hgb  = st.number_input("Hemoglobin HGB (g/dL)", 3.0, 20.0, float(D['HGB']), 0.1,
                                   help="WHO threshold: <11.0 g/dL = Anemic")
            rbc  = st.number_input("RBC (10¹²/L)",          1.0, 8.0,  float(D['RBC']), 0.1)
            hct  = st.number_input("Hematocrit HCT (%)",   10.0, 60.0, float(D['HCT']), 0.5)
            mcv  = st.number_input("MCV (fL)",             40.0, 130.0, float(D['MCV']), 0.5)
        with c2:
            wbc  = st.number_input("WBC (10⁹/L)",          2.0, 30.0,  float(D['WBC']), 0.1)
            plt_ = st.number_input("Platelets PLT (10⁹/L)",50.0,800.0, float(D['PLT']), 5.0)
            mch  = st.number_input("MCH (pg)",             15.0, 45.0,  float(D['MCH']), 0.5)
            rdw  = st.number_input("RDW (%)",               8.0, 30.0,  float(D['RDW']), 0.1)

        with st.expander("⚙️ Additional Maternal Vitals"):
            c3, c4 = st.columns(2)
            with c3:
                age  = st.number_input("Age (years)",        14.0, 55.0, float(D['Age']),  1.0)
                bmi  = st.number_input("BMI (kg/m²)",        12.0, 50.0, float(D['BMI']),  0.5)
                bs   = st.number_input("Blood Sugar (mmol/L)", 2.0, 25.0, float(D['BS']),   0.1)
                temp = st.number_input("Body Temp (°F)",     95.0,105.0, float(D['Body Temp']), 0.2)
            with c4:
                sbp  = st.number_input("Systolic BP (mmHg)", 60.0,220.0, float(D['Systolic BP']),  1.0)
                dbp  = st.number_input("Diastolic BP (mmHg)",40.0,140.0, float(D['Diastolic']),    1.0)
                hr   = st.number_input("Heart Rate (bpm)",   40.0,160.0, float(D['Heart Rate']),   1.0)
                mchc = st.number_input("MCHC (g/dL)",        20.0, 45.0, float(D['MCHC']), 0.5)
                mpv  = st.number_input("MPV (fL)",            5.0, 20.0, float(D['MPV']),  0.2)

        patient_df = pd.DataFrame([{
            'Age': age, 'BMI': bmi, 'BS': bs, 'Body Temp': temp,
            'Diastolic': dbp, 'HCT': hct, 'HGB': hgb, 'Heart Rate': hr,
            'MCH': mch, 'MCHC': mchc, 'MCV': mcv, 'MPV': mpv,
            'PLT': plt_, 'RBC': rbc, 'RDW': rdw, 'Systolic BP': sbp, 'WBC': wbc
        }])[FEATURES]

    # ── Image Input ───────────────────────────────────────────────────────
    with col_right:
        st.subheader("2️⃣ Non-Invasive Image Biometric")
        st.markdown("Upload a **palpebral conjunctiva** (lower eyelid) or **nail bed** photograph.")
        uploaded = st.file_uploader("Choose image (JPG/PNG)", type=["jpg","jpeg","png"])
        pil_img  = None
        if uploaded:
            try:
                pil_img = Image.open(uploaded).convert("RGB")
                st.image(pil_img, caption="Uploaded biometric image", use_column_width=True)
            except Exception as e:
                st.error(f"Cannot open image: {e}")
        else:
            st.info("No image uploaded — prediction will rely only on clinical data.")

    st.divider()

    # ── Prediction ────────────────────────────────────────────────────────
    st.subheader("3️⃣ Multimodal Fusion Diagnosis")

    if st.button("🚀 Run Multimodal Screening", type="primary", use_container_width=True):
        # --- Tabular prediction ---
        if tab_model:
            tab_prob = float(tab_model.predict_proba(patient_df)[0, 1])
        else:
            tab_prob = 1.0 if hgb < 11.0 else 0.05  # fallback: WHO rule

        tab_label = "Anemic" if tab_prob >= 0.5 else "Non-Anemic"

        # --- Image prediction ---
        img_prob  = None
        img_label = None
        if pil_img and img_model and device:
            try:
                tensor = image_transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(img_model(tensor), dim=1).cpu().numpy()[0]
                img_prob  = float(probs[1])
                img_label = "Anemic" if img_prob >= 0.5 else "Non-Anemic"
            except Exception as e:
                st.warning(f"Image inference skipped: {e}")

        # --- Late fusion ---
        if img_prob is not None:
            fused = 0.60 * tab_prob + 0.40 * img_prob
            mode  = "Multimodal (CBC + Image)"
        else:
            fused = tab_prob
            mode  = "Single modality (CBC only)"

        is_anemic = fused >= 0.5

        # ── Display metric cards ──────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div>Clinical CBC (XGBoost)</div>
                <div class="metric-val" style="color:#1f77b4">{tab_prob:.1%}</div>
                <div>{'⚠️ ' if tab_label=='Anemic' else '✅ '}{tab_label}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            if img_prob is not None:
                st.markdown(f"""
                <div class="metric-box">
                    <div>Visual Biometric (ResNet-18)</div>
                    <div class="metric-val" style="color:#9467bd">{img_prob:.1%}</div>
                    <div>{'⚠️ ' if img_label=='Anemic' else '✅ '}{img_label}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="metric-box"><div>Visual Biometric</div>
                    <div class="metric-val" style="color:#666">—</div>
                    <div>No image provided</div></div>""", unsafe_allow_html=True)
        with m3:
            color = "#e74c3c" if is_anemic else "#2ecc71"
            st.markdown(f"""
            <div class="metric-box">
                <div>Fused Risk Score ({mode})</div>
                <div class="metric-val" style="color:{color}">{fused:.1%}</div>
                <div>{'⚠️ HIGH RISK' if is_anemic else '✅ LOW RISK'}</div>
            </div>""", unsafe_allow_html=True)

        st.write("")

        # ── Clinical Alert ────────────────────────────────────────────────
        if is_anemic:
            if hgb < 7.0:
                sev   = "SEVERE PRENATAL ANEMIA"
                advice= "🚨 EMERGENCY: Go to hospital immediately. IV iron or blood transfusion may be required."
            elif hgb < 10.0:
                sev   = "MODERATE PRENATAL ANEMIA"
                advice= "⚠️ HIGH PRIORITY: See your obstetrician within 24–48 hours. High-dose iron therapy required."
            else:
                sev   = "MILD PRENATAL ANEMIA"
                advice= "📋 MONITOR: Consult your doctor. Start oral iron + Vitamin C and improve your diet."
            st.markdown(f'<div class="risk-high">⚠️ {sev} &nbsp;|&nbsp; Risk: {fused:.1%}<br><small>{advice}</small></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-ok">✅ Non-Anemic / Normal &nbsp;|&nbsp; Risk: {fused:.1%}<br>'
                        '<small>Continue standard antenatal care and maintain a balanced diet.</small></div>',
                        unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — HEALTH & DIET GUIDELINES
# ═════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.header("🥗 Prenatal Anemia — Health & Dietary Guide")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("🚨 When to See a Doctor Immediately")
        st.markdown("""
| Sign | Action |
|---|---|
| HGB < 7.0 g/dL (Severe) | **Emergency hospital admission** |
| HGB 7–10 g/dL (Moderate) | Doctor visit within **24–48 hours** |
| HGB 10–11 g/dL (Mild) | Routine obstetric review within **7 days** |
| HGB ≥ 11.0 g/dL (Normal) | Continue standard prenatal care |

**Seek emergency care if you experience:**
- Fainting or severe dizziness
- Rapid or pounding heartbeat at rest
- Shortness of breath while resting
- Extreme pallor of lips, tongue, eyelids, or nails
- Unusual cravings for non-food items (ice, clay, dirt) — a sign called **Pica**
        """)

        st.subheader("📊 WHO Anemia Classification")
        st.markdown("""
| Tier | Hemoglobin | Clinical Stage |
|---|---|---|
| Normal | ≥ 11.0 g/dL | Non-Anemic |
| Mild | 10.0 – 10.9 g/dL | Monitor + Supplement |
| Moderate | 7.0 – 9.9 g/dL | Therapeutic Iron |
| Severe | < 7.0 g/dL | Emergency |
        """)

    with c2:
        st.subheader("🥦 Iron-Rich Foods to Eat")
        st.markdown("""
**🥩 Heme Iron (highest absorption 15–35%):**
- Lean beef, lamb, chicken, turkey
- Tuna, salmon, sardines

**🌿 Non-Heme Iron (plant-based, 2–20% absorption):**
- Spinach, kale, Swiss chard, broccoli
- Lentils, chickpeas, kidney beans, black beans
- Pumpkin seeds, sesame, almonds
- Fortified cereals and oatmeal

**💊 Vitamin C — Iron Absorption Booster:**
Always eat these *with* your iron-rich plant foods to triple absorption:
- Oranges, lemons, kiwi, strawberries
- Bell peppers (especially red/yellow), tomatoes

**❌ What to Avoid Around Meal Times:**
- ☕ Tea or coffee (tannins block absorption)
- 🥛 Milk or dairy near iron meal (calcium competes)
- Wait at least **2 hours** after iron-rich meals
        """)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
with tab_paper:
    st.header("📊 Trained Model Performance")
    st.markdown("Results from training on **103,221 CBC records** + **7,041 conjunctival/nail images**.")

    # Load saved metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            met = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tabular Accuracy",   f"{met['summary']['tabular_accuracy']:.2%}")
        m2.metric("Tabular AUC-ROC",    f"{met['summary']['tabular_auc']:.4f}")
        m3.metric("Image Accuracy",     f"{met['summary']['image_accuracy']:.2%}")
        m4.metric("Image AUC-ROC",      f"{met['summary']['image_auc']:.4f}")

    st.divider()

    col1, col2 = st.columns(2)
    figs = {
        "Confusion Matrix (ResNet-18)":  os.path.join(FIGURES_DIR, "img_confusion_matrix.png"),
        "Model Comparison":              os.path.join(FIGURES_DIR, "model_comparison.png"),
        "Feature Importance (XGBoost)":  os.path.join(FIGURES_DIR, "feature_importance.png"),
        "ROC Curves (All Models)":       os.path.join(FIGURES_DIR, "roc_curves.png"),
    }
    for i, (title, path) in enumerate(figs.items()):
        target = col1 if i % 2 == 0 else col2
        if os.path.exists(path):
            target.image(path, caption=title, use_column_width=True)
        else:
            target.warning(f"{title} not available.")

    st.divider()
    st.markdown("""
**Architecture Summary:**

| Component | Details |
|---|---|
| Tabular Model | XGBoost — 300 trees, depth=5, lr=0.05 |
| Image Model | ResNet-18 fine-tuned on ImageNet |
| Fusion | Confidence-weighted late fusion (α=0.60 tabular, α=0.40 image) |
| Training Data | 103,221 CBC records + 1,179 maternal records + 7,041 images |
| Label Criterion | WHO threshold: HGB < 11.0 g/dL = Anemic |
    """)
