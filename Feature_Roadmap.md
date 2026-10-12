# 📈 Enhanced Feature Roadmap for Prenatal Anemia AI Prototype

Below is a **prioritized roadmap** that aligns the Streamlit interface with the research proposal and adds high‑impact research‑oriented capabilities.  Each entry lists the feature, its purpose, required implementation effort, and any dependencies.

---

## 🎯 Top‑Priority (Must‑Have) – ✅
| # | Feature | Why it matters | Implementation notes |
|---|---------|----------------|----------------------|
| 1 | **Transformer‑Based Multimodal Fusion** | Aligns the model architecture with the paper (tabular encoder + image encoder → transformer → severity prediction). | Replace current `FusionModel` with a `nn.TransformerEncoder` that consumes the 64‑dim tabular embedding and 512‑dim image embedding.  Train end‑to‑end; output 4‑class severity.  Requires updating `src/models.py` and re‑training (`src/train.py`). |
| 2 | **AI Explanation Panel** | Provides scientific transparency (required for a 2026 AI research demo). | • Tabular: compute SHAP values (`shap.TreeExplainer` for XGBoost) → bar plot.
• Image: generate Grad‑CAM heatmap (`torchvision.transforms`, `torch.nn.functional.grad_cam`).
Display side‑by‑side in Streamlit with `st.pyplot`/`st.image`.
Add a new Streamlit tab *"Explainability"* after prediction. |
| 3 | **Multimodal Confidence & Agreement Visual** | Shows how each modality contributes to the final decision, highlights disagreements. | After inference, collect class probabilities from tabular, image, and fusion models. Render a horizontal bar (e.g., `st.progress` or custom `altair` chart) and a textual agreement badge (`✅`/`⚠️`). |

---
## 🚀 High‑Impact Add‑Ons – 🟡
| # | Feature | Reason | Effort |
|---|---------|--------|-------|
| 4 | **Image Quality Checker** | Prevents garbage‑in predictions on low‑quality smartphone captures. | Use OpenCV to compute brightness, blur (`cv2.Laplacian` variance), resolution, and eye‑region detection (Haar cascade or a tiny CNN). Show ✅/❌ checklist before enabling *Run Screening*. |
| 5 | **What‑If Simulator** | Demonstrates model sensitivity and helps users understand variable impact. | Provide UI sliders for key tabular features; on *Simulate* recompute prediction using only the tabular encoder (fast) and display delta risk. |
| 6 | **Cross‑Population Validation Dashboard** | Shows external generalisation (NHANES, DHS, etc.) – essential for a research paper. | Add a new *"Validation"* tab with a table of datasets and metrics (accuracy, recall, F1, AUC). Load numbers from `results/validation_metrics.json` (pre‑computed). |
| 7 | **Interactive Ablation Study** | Lets examiners toggle modalities and instantly see performance changes. | Checkbox group (Tabular, Image, Fusion) → load pre‑saved metrics for each configuration (from `results/ablation.json`) and display a concise table/plot. |
| 8 | **Automatic PDF Screening Report** | Produces a professional, printable summary for clinicians or reviewers. | Use `reportlab` or `fpdf` to assemble patient ID, inputs, modality probabilities, SHAP bar, Grad‑CAM thumbnail, and recommendations. Add a *"Download Report"* button. |
| 9 | **Clinical Triage Timeline** | Converts risk scores into a clear care pathway (required for regulatory transparency). | Visual flowchart (Mermaid `flowchart LR`) showing *Result → Risk Level → Urgency → Recommended Action → Confirmatory Test*. |
|10| **Privacy Mode** | Aligns with NIST AI RMF and GDPR‑style safeguards. | Add a toggle; when ON, do not write uploaded images to disk, generate a random session token, and display a short privacy notice. |
|11| **Patient Risk Profile Gauge** | Summarises multimodal contributions at a glance. | Combine three horizontal bars (Iron status, Clinical factors, Image signal) using `st.progress` with custom colors. |
|12| **Evidence / Research Mode** | Links predictions back to datasets, model version, and literature. | New tab *"Evidence"* that reads metadata from `results/model_info.json` and lists the 10 cited papers with clickable links. |

---
## 📦 Low‑Effort Cosmetic Enhancements – 🔵
| # | Feature | Quick win |
|---|---------|----------|
| A | Dark theme & custom CSS (already in `.streamlit/config.toml`). |
| B | Add animated loading spinners during inference (`st.spinner`). |
| C | Show model version badge (e.g., `v1.2‑TransformerFusion`). |

---
## 📂 Where to Implement
- **Model code** – `src/models.py` (add `TabularEncoder`, `ImageEncoder`, `TransformerFusion`).
- **Training script** – `src/train.py` (update to train the transformer‑fusion end‑to‑end). 
- **Streamlit UI** – `app.py` (new tabs: *Explainability*, *Validation*, *Evidence*; new components for image‑quality, what‑if, PDF report). 
- **Artifacts** – store pre‑computed SHAP values (`results/shap_values.pkl`), Grad‑CAM images (`results/grad_cam/`), and validation metrics (`results/validation_metrics.json`).

---
## 📅 Suggested Development Timeline (2‑week sprint)
| Day | Goal |
|-----|------|
| 1‑2 | Refactor `src/models.py` → implement transformer fusion; update config. |
| 3‑4 | Re‑train model; dump new checkpoint `fusion_transformer.pt`. |
| 5   | Integrate SHAP & Grad‑CAM utilities; generate example visual assets. |
| 6‑7 | Extend `app.py` with Explanation panel and Confidence‑Agreement bar. |
| 8   | Add Image Quality checker (OpenCV). |
| 9   | Build What‑If simulator (tabular sliders only). |
|10   | Create Validation dashboard & Ablation UI. |
|11   | Implement PDF report generation. |
|12   | Polish UI, add privacy toggle, final testing on Streamlit Cloud. |

---
### ✅ Next Steps for You
1. **Approve the roadmap** – let me know which features to prioritize (e.g., top‑3). 
2. **Confirm model training resources** – we need a GPU‑enabled runner for the transformer‑fusion training (could be a local machine or a cloud instance). 
3. **Give any style preferences** for the PDF report (logo, colors). 

Feel free to ask for any additional details or to start coding any of the items above! 🎉
