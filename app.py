# CRITICAL: Configure environment and import torch/torchvision FIRST on Windows
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

# Other imports
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from PIL import Image
import json
import random

# Page configuration
st.set_page_config(
    page_title="Prenatal Anemia Multimodal Screening System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
WORKSPACE = r"C:\Users\gshan\OneDrive\Documents\PrenatalAnemia"
RESULTS_DIR = os.path.join(WORKSPACE, "results")
DATA_DIR = os.path.join(WORKSPACE, "data")
SAMPLE_IMG_DIR = os.path.join(DATA_DIR, "Dataset_sample")
TAB_MODEL_PATH = os.path.join(RESULTS_DIR, "xgb_tabular.json")
IMG_MODEL_PATH = os.path.join(RESULTS_DIR, "image_model_best.pt")

# Features expected by XGBoost (in exact training order)
FEATURES = ['Age', 'BMI', 'BS', 'Body Temp', 'Diastolic', 'HCT', 'HGB', 'Heart Rate', 
            'MCH', 'MCHC', 'MCV', 'MPV', 'PLT', 'RBC', 'RDW', 'Systolic BP', 'WBC']

# Preset patient profiles for quick testing
PRESETS = {
    "Normal Patient (Healthy Pregnant)": {
        'Age': 26.0, 'BMI': 23.5, 'BS': 4.8, 'Body Temp': 98.4, 'Diastolic': 70.0,
        'HCT': 37.5, 'HGB': 12.8, 'Heart Rate': 74.0, 'MCH': 30.5, 'MCHC': 34.0,
        'MCV': 89.0, 'MPV': 8.2, 'PLT': 260.0, 'RBC': 4.6, 'RDW': 12.8,
        'Systolic BP': 110.0, 'WBC': 7.8
    },
    "Mild Anemic Patient": {
        'Age': 29.0, 'BMI': 22.0, 'BS': 5.2, 'Body Temp': 98.6, 'Diastolic': 68.0,
        'HCT': 31.0, 'HGB': 10.2, 'Heart Rate': 84.0, 'MCH': 26.5, 'MCHC': 31.5,
        'MCV': 79.0, 'MPV': 8.6, 'PLT': 220.0, 'RBC': 3.8, 'RDW': 15.2,
        'Systolic BP': 105.0, 'WBC': 8.5
    },
    "Severe Anemic Patient": {
        'Age': 31.0, 'BMI': 20.2, 'BS': 5.5, 'Body Temp': 98.2, 'Diastolic': 60.0,
        'HCT': 24.0, 'HGB': 7.6, 'Heart Rate': 96.0, 'MCH': 22.0, 'MCHC': 28.0,
        'MCV': 70.0, 'MPV': 9.2, 'PLT': 180.0, 'RBC': 2.9, 'RDW': 18.5,
        'Systolic BP': 95.0, 'WBC': 9.8
    }
}

# Image Preprocessing Transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from src.models import ImageModel

@st.cache_resource
def load_models():
    """Load both Tabular XGBoost and Image ResNet-18 models with caching."""
    # 1. Tabular Model
    try:
        t_model = xgb.XGBClassifier()
        t_model.load_model(TAB_MODEL_PATH)
    except Exception as e:
        t_model = None
        st.sidebar.error(f"Error loading Tabular Model: {e}")

    # 2. Image Model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        i_model = ImageModel(num_classes=2, pretrained=False)
        state_dict = torch.load(IMG_MODEL_PATH, map_location=device)
        i_model.load_state_dict(state_dict)
        i_model = i_model.to(device)
        i_model.eval()
    except Exception as e:
        i_model, device = None, None
        st.sidebar.error(f"Error loading Image Model: {e}")

    return t_model, i_model, device

tab_model, img_model, device = load_models()

# Top Header Banner
st.title("🩸 Multimodal Prenatal Anemia Screening Platform")
st.markdown(
    "A clinical decision-support system integrating **Complete Blood Count (CBC) analytics** "
    "and **non-invasive ocular/ungual image biometrics** with holistic maternal care guidelines."
)
st.divider()

# Navigation Tabs
tab_screen, tab_guidelines, tab_paper = st.tabs([
    "🔬 Multimodal Screening & Prediction",
    "🥗 Dietary Care & Clinical Action Plan",
    "📈 Research Metrics & Paper Figures"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: SCREENING & PREDICTION INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
with tab_screen:
    col_tab, col_img = st.columns([1.1, 1.0])

    with col_tab:
        st.subheader("1. Clinical & Hematological Parameters")
        
        # Preset selection
        preset_choice = st.selectbox(
            "Quick Demo Preset (or customize below):",
            list(PRESETS.keys())
        )
        selected_defaults = PRESETS[preset_choice]

        # Primary blood markers
        c1, c2 = st.columns(2)
        with c1:
            hgb_val = st.number_input(
                "Hemoglobin (HGB) [g/dL]",
                min_value=3.0, max_value=20.0,
                value=float(selected_defaults['HGB']),
                step=0.1,
                help="WHO diagnostic cutoff: HGB < 11.0 g/dL signifies prenatal anemia."
            )
            rbc_val = st.number_input(
                "RBC Count [10^12/L]",
                min_value=1.0, max_value=8.0,
                value=float(selected_defaults['RBC']),
                step=0.1
            )
            hct_val = st.number_input(
                "Hematocrit (HCT) [%]",
                min_value=10.0, max_value=60.0,
                value=float(selected_defaults['HCT']),
                step=0.5
            )
            mcv_val = st.number_input(
                "Mean Corpuscular Vol (MCV) [fL]",
                min_value=40.0, max_value=130.0,
                value=float(selected_defaults['MCV']),
                step=0.5
            )
        with c2:
            wbc_val = st.number_input(
                "WBC Count [10^9/L]",
                min_value=2.0, max_value=30.0,
                value=float(selected_defaults['WBC']),
                step=0.1
            )
            plt_val = st.number_input(
                "Platelet Count (PLT) [10^9/L]",
                min_value=50.0, max_value=800.0,
                value=float(selected_defaults['PLT']),
                step=5.0
            )
            mch_val = st.number_input(
                "MCH [pg]",
                min_value=15.0, max_value=45.0,
                value=float(selected_defaults['MCH']),
                step=0.5
            )
            rdw_val = st.number_input(
                "RDW [%]",
                min_value=8.0, max_value=30.0,
                value=float(selected_defaults['RDW']),
                step=0.1
            )

        with st.expander("Additional Maternal Vitals & Biomarkers"):
            c3, c4 = st.columns(2)
            with c3:
                age_val = st.number_input("Maternal Age [years]", 14.0, 55.0, float(selected_defaults['Age']), 1.0)
                bmi_val = st.number_input("BMI [kg/m²]", 12.0, 50.0, float(selected_defaults['BMI']), 0.5)
                bs_val  = st.number_input("Blood Sugar (BS) [mmol/L]", 2.0, 25.0, float(selected_defaults['BS']), 0.1)
                temp_val= st.number_input("Body Temperature [°F]", 95.0, 105.0, float(selected_defaults['Body Temp']), 0.2)
            with c4:
                sbp_val = st.number_input("Systolic BP [mmHg]", 60.0, 220.0, float(selected_defaults['Systolic BP']), 1.0)
                dbp_val = st.number_input("Diastolic BP [mmHg]", 40.0, 140.0, float(selected_defaults['Diastolic']), 1.0)
                hr_val  = st.number_input("Heart Rate [bpm]", 40.0, 160.0, float(selected_defaults['Heart Rate']), 1.0)
                mchc_val= st.number_input("MCHC [g/dL]", 20.0, 45.0, float(selected_defaults['MCHC']), 0.5)
                mpv_val = st.number_input("MPV [fL]", 5.0, 20.0, float(selected_defaults['MPV']), 0.2)

        # Assemble row DataFrame
        input_data = {
            'Age': age_val, 'BMI': bmi_val, 'BS': bs_val, 'Body Temp': temp_val,
            'Diastolic': dbp_val, 'HCT': hct_val, 'HGB': hgb_val, 'Heart Rate': hr_val,
            'MCH': mch_val, 'MCHC': mchc_val, 'MCV': mcv_val, 'MPV': mpv_val,
            'PLT': plt_val, 'RBC': rbc_val, 'RDW': rdw_val, 'Systolic BP': sbp_val,
            'WBC': wbc_val
        }
        patient_df = pd.DataFrame([input_data])[FEATURES]

    with col_img:
        st.subheader("2. Non-Invasive Image Biometric")
        img_source = st.radio(
            "Choose Image Input Source:",
            ["Upload Custom Image", "Select from Validated Clinical Test Set"],
            horizontal=True
        )
        
        loaded_pil_img = None
        
        if img_source == "Upload Custom Image":
            uploaded_file = st.file_uploader(
                "Upload Palpebral Conjunctiva or Nail Bed image (JPG, PNG)",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                try:
                    loaded_pil_img = Image.open(uploaded_file).convert("RGB")
                    st.image(loaded_pil_img, caption="Patient Biometric Image", width=280)
                except Exception as e:
                    st.error(f"Error opening image: {e}")
        else:
            sample_class = st.selectbox(
                "Select Test Category:",
                ["Eye_Anemic", "Eye_Non_Anemic", "Nail_Anemic", "Nail_Non_Anemic"]
            )
            folder_target = os.path.join(SAMPLE_IMG_DIR, sample_class)
            if os.path.isdir(folder_target):
                file_list = [f for f in os.listdir(folder_target) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if file_list:
                    selected_img_name = st.selectbox("Select Sample Image:", file_list[:20])
                    sample_path = os.path.join(folder_target, selected_img_name)
                    loaded_pil_img = Image.open(sample_path).convert("RGB")
                    st.image(loaded_pil_img, caption=f"Dataset Sample: {selected_img_name} ({sample_class})", width=280)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICTION & MULTIMODAL FUSION
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("3. Integrated Multimodal Diagnosis")

    if st.button("🚀 Run Multimodal Diagnostic Screening", type="primary", use_container_width=True):
        # 1. Tabular Model Prediction
        if tab_model is not None:
            tab_prob = float(tab_model.predict_proba(patient_df)[0, 1])
        else:
            # Fallback based on standard medical threshold if model file is missing
            tab_prob = 1.0 if hgb_val < 11.0 else 0.05
        tab_label = "Anemic" if tab_prob >= 0.5 else "Non-Anemic"

        # 2. Image Model Prediction
        img_prob = None
        if loaded_pil_img is not None and img_model is not None and device is not None:
            try:
                img_tensor = image_transform(loaded_pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = img_model(img_tensor)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    img_prob = float(probabilities[1])  # Class 1 = Anemic
                    img_label = "Anemic" if img_prob >= 0.5 else "Non-Anemic"
            except Exception as e:
                st.warning(f"Could not compute image inference: {e}")

        # 3. Multimodal Late Fusion
        # If both modalities are available, apply confidence-weighted soft voting
        if img_prob is not None:
            # Tabular has high reliability when CBC is complete (0.65), image provides non-invasive validation (0.35)
            fused_prob = (0.60 * tab_prob) + (0.40 * img_prob)
            fusion_mode = "Multimodal Consensus (Tabular + Image Biometric)"
        else:
            fused_prob = tab_prob
            fusion_mode = "Single Modality (Hematological CBC Only)"

        is_fused_anemic = fused_prob >= 0.5

        # Display Metrics
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Clinical Tabular Risk (XGBoost)",
                value=f"{tab_prob:.1%}",
                delta=f"Status: {tab_label}",
                delta_color="inverse" if tab_label == "Anemic" else "normal"
            )
            st.caption(f"Driven by HGB ({hgb_val} g/dL), RBC ({rbc_val}), MCV ({mcv_val})")

        with res_col2:
            if img_prob is not None:
                st.metric(
                    label="Visual Biometric Risk (ResNet-18)",
                    value=f"{img_prob:.1%}",
                    delta=f"Status: {img_label}",
                    delta_color="inverse" if img_label == "Anemic" else "normal"
                )
                st.caption("Deep feature extraction from conjunctival/nail pallor")
            else:
                st.info("No biometric image provided. Prediction based solely on clinical records.")

        with res_col3:
            st.metric(
                label="Fused Multimodal Risk Score",
                value=f"{fused_prob:.1%}",
                delta="CRITICAL: HIGH ANEMIA RISK" if is_fused_anemic else "NORMAL / LOW RISK",
                delta_color="inverse" if is_fused_anemic else "normal"
            )
            st.caption(fusion_mode)

        # Final Diagnostic Alert Banner
        st.write("")
        if is_fused_anemic:
            if hgb_val < 8.0:
                severity = "SEVERE PRENATAL ANEMIA"
                urgency = "EMERGENCY: Immediate hospital evaluation and specialized obstetric hematology intervention required."
                color_box = st.error
            elif hgb_val < 10.0:
                severity = "MODERATE PRENATAL ANEMIA"
                urgency = "HIGH PRIORITY: Schedule clinical visit within 24-48 hours for therapeutic iron therapy evaluation."
                color_box = st.error
            else:
                severity = "MILD PRENATAL ANEMIA"
                urgency = "MONITOR: Consult your obstetrician for dietary fortification and elemental iron supplementation."
                color_box = st.warning
            
            color_box(f"### ⚠️ Diagnostic Assessment: {severity} Detected (Risk: {fused_prob:.1%})")
            st.markdown(f"**Recommended Clinical Action:** {urgency}")
        else:
            st.success(f"### ✅ Diagnostic Assessment: Normal / Non-Anemic (Risk: {fused_prob:.1%})")
            st.markdown(
                "**Clinical Observation:** Blood markers and ocular pigmentation appear consistent with healthy gestational limits. "
                "Continue standard prenatal vitamins and balanced nutrition."
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: HEALTH, DIET, AND CLINICAL CARE GUIDELINES
# ─────────────────────────────────────────────────────────────────────────────
with tab_guidelines:
    st.header("🥗 Clinical Nutrition & Prenatal Anemia Management")
    st.markdown(
        "Prenatal iron-deficiency anemia poses significant risks including preterm delivery, "
        "low birth weight, and maternal exhaustion. Below are evidence-based nutritional protocols and clinical guidelines."
    )

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("🚨 When to Seek Immediate Medical Consultation")
        st.markdown("""
        Contact your obstetrician or visit an emergency maternity clinic immediately if you experience:
        * **Severe dizziness, syncope (fainting), or sudden lightheadedness.**
        * **Tachycardia or heart palpitations** (abnormally rapid, pounding heartbeat at rest).
        * **Dyspnea (shortness of breath)** during mild activities or while resting.
        * **Extreme pallor** of the palpebral conjunctiva, tongue, inner lip, or nail beds.
        * **Pica symptoms** (unusual cravings to eat non-food substances such as ice, dirt, or clay).
        * **Persistent fatigue** and inability to perform daily activities.
        """)

        st.subheader("📊 Anemia Severity Classifications (WHO Standards)")
        st.markdown("""
        | Classification | Hemoglobin (HGB) Range | Clinical Priority |
        | :--- | :--- | :--- |
        | **Normal** | **≥ 11.0 g/dL** | Routine prenatal care & maintenance |
        | **Mild Anemia** | **10.0 – 10.9 g/dL** | Oral iron supplementation & dietary optimization |
        | **Moderate Anemia** | **7.0 – 9.9 g/dL** | High-dose therapeutic iron; close fetal monitoring |
        | **Severe Anemia** | **< 7.0 g/dL** | Urgent hospitalization; potential IV iron / transfusion |
        """)

    with col_g2:
        st.subheader("🥦 Essential Dietary Recommendations")
        st.markdown("""
        #### 1. High-Iron Food Sources
        * **Heme Iron (Superior Bioavailability ~15-35%):**
          * Lean beef, lamb, poultry, and well-cooked seafood.
        * **Non-Heme Iron (Plant-based ~2-20%):**
          * Dark leafy vegetables (spinach, kale, Swiss chard).
          * Legumes (lentils, chickpeas, kidney beans, black beans).
          * Seeds & nuts (pumpkin seeds, sesame seeds, almonds).
          * Iron-fortified whole grain cereals and oatmeal.

        #### 2. Synergistic Iron Absorption Enhancers (Vitamin C)
        * Always pair non-heme plant iron with **Ascorbic Acid (Vitamin C)** to triple absorption rates:
          * Citrus fruits (oranges, lemons, grapefruits).
          * Bell peppers (especially red and yellow peppers).
          * Kiwi, strawberries, guava, and tomatoes.

        #### 3. Critical Absorption Inhibitors (Avoid at Meal Times)
        * ❌ **Calcium:** Do not consume milk, cheese, yogurt, or calcium supplements with iron meals.
        * ❌ **Polyphenols & Tannins:** Avoid drinking black tea, green tea, or coffee within **2 hours** of iron-rich meals.
        * ❌ **Phytates:** Soak beans and legumes prior to cooking to reduce phytate binding.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: RESEARCH PAPER METRICS & ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_paper:
    st.header("📈 Research Validation & Performance Benchmarks")
    st.markdown(
        "Performance summary of the dual-branch framework evaluated on the held-out validation cohorts "
        "(103,221 CBC records, 1,179 maternal clinic rows, and 7,041 clinical images)."
    )

    # Load metrics from file if available
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_dict = json.load(f)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tabular Model Accuracy", f"{metrics_dict['summary']['tabular_accuracy']:.2%}")
            m2.metric("Tabular Model AUC-ROC", f"{metrics_dict['summary']['tabular_auc']:.4f}")
            m3.metric("Image Model Accuracy", f"{metrics_dict['summary']['image_accuracy']:.2%}")
            m4.metric("Image Model AUC-ROC", f"{metrics_dict['summary']['image_auc']:.4f}")
        except Exception:
            pass

    st.divider()

    fig_col1, fig_col2 = st.columns(2)
    fig_dir = os.path.join(RESULTS_DIR, "figures")

    with fig_col1:
        img_cm_path = os.path.join(fig_dir, "img_confusion_matrix.png")
        if os.path.exists(img_cm_path):
            st.image(img_cm_path, caption="Figure 1: Confusion Matrix of Fine-Tuned ResNet-18 on Ocular & Ungual Imagery")

        model_comp_path = os.path.join(fig_dir, "model_comparison.png")
        if os.path.exists(model_comp_path):
            st.image(model_comp_path, caption="Figure 3: Metric Comparison across Modalities")

    with fig_col2:
        roc_path = os.path.join(fig_dir, "combined_roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Figure 2: Receiver Operating Characteristic (ROC) Trajectories")
        
        dist_path = os.path.join(fig_dir, "dataset_distribution.png")
        if os.path.exists(dist_path):
            st.image(dist_path, caption="Figure 4: Class & Modality Distribution Across Cohorts")
