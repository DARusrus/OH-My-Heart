<div align="center">

<!-- ANIMATED HEADER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=🫀%20CardioRisk%20AI&fontSize=52&fontColor=fff&animation=fadeIn&fontAlignY=35&desc=Clinical-Grade%20Cardiovascular%20Disease%20Classifier&descAlignY=58&descSize=16&descColor=aad4f5" width="100%"/>

<!-- BADGES ROW 1 -->
<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Tuned-E34234?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-Tuned-9AC122?style=for-the-badge&logo=lightgbm&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Gradio-5.29.0-FF7C00?style=for-the-badge&logo=gradio&logoColor=white"/>
</p>

<!-- BADGES ROW 2 -->
<p>
  <img src="https://img.shields.io/badge/ROC--AUC-0.802-success?style=flat-square&logo=chartdotjs"/>
  <img src="https://img.shields.io/badge/Dataset-68%2C634%20Patients-blue?style=flat-square&logo=kaggle"/>
  <img src="https://img.shields.io/badge/Validation-5--Fold%20Stratified%20CV-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Calibration-Isotonic%20Regression-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-gray?style=flat-square"/>
</p>

<!-- HERO STATEMENT -->
<br/>

> ### *"A machine learning system that predicts cardiovascular disease risk*
> ### *and explains every prediction — the way a clinician would."*

<br/>

<!-- LIVE DEMO BUTTON -->
<a href="#">
  <img src="https://img.shields.io/badge/🚀%20%20LIVE%20DEMO%20%20→%20%20Try%20it%20Now-FF4B4B?style=for-the-badge&logoColor=white" height="42"/>
</a>
&nbsp;&nbsp;
<a href="#">
  <img src="https://img.shields.io/badge/📓%20%20Kaggle%20Notebook%20%20→%20%20Open-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" height="42"/>
</a>

<br/><br/>

</div>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 1 — OVERVIEW -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 📋 &nbsp; Project Overview

</div>

**CardioRisk AI** is a full-stack machine learning project that classifies cardiovascular disease risk from routine clinical measurements. It goes beyond a simple accuracy score — it is **calibrated**, **explained**, **cross-validated**, and **deployed** as an interactive clinical tool.

Built as a portfolio project for a Computer Science degree, it demonstrates the complete ML pipeline from raw data to production-grade deployment, applying techniques used in real clinical decision support systems.

<br/>

<div align="center">

<!-- PIPELINE FLOW DIAGRAM -->

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🫀  ML PIPELINE OVERVIEW                            ║
╠══════════╦══════════════╦══════════════╦══════════════╦══════════════════════╣
║  📦 DATA ║ 🔬 ENGINEER  ║  🤖 TRAIN   ║  📊 EXPLAIN  ║    🚀 DEPLOY        ║
║          ║              ║              ║              ║                      ║
║ 68,634   ║ Clinical FE  ║ XGBoost +    ║ SHAP Tree-   ║ Gradio App +         ║
║ patients ║ JNC-8 BP     ║ LightGBM     ║ Explainer    ║ Live Public URL      ║
║ 11 raw   ║ WHO BMI      ║ Optuna Tune  ║ Beeswarm     ║ 11 inputs →          ║
║ features ║ MAP, PP      ║ 5-Fold CV    ║ Waterfall    ║ Risk score +         ║
║ cleaned  ║ Interactions ║ Ensemble     ║ Dependence   ║ SHAP plot            ║
║ outliers ║ 20 features  ║ Calibrated   ║ Calibration  ║                      ║
╚══════════╩══════════════╩══════════════╩══════════════╩══════════════════════╝
```

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 2 — RESULTS -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 🏆 &nbsp; Model Performance

<br/>

### Final Results — Held-Out Test Set (13,727 patients)

<br/>

|  | Model | ROC-AUC | F1 Score | Accuracy | Avg Precision |
|:-:|:------|:-------:|:--------:|:--------:|:-------------:|
| 🥇 | **LightGBM (Tuned + Ensemble)** | **0.8021** | **0.7211** | **0.7341** | **0.7843** |
| 🥈 | XGBoost (Tuned) | 0.8017 | 0.7193 | 0.7333 | 0.7834 |
| 🥉 | LightGBM (Tuned) | 0.8013 | 0.7205 | 0.7331 | 0.7835 |
| — | Soft Ensemble (5 models) | 0.8020 | 0.7184 | 0.7326 | 0.7844 |
| — | XGBoost (Default) | 0.8003 | 0.7189 | 0.7327 | 0.7816 |
| — | Logistic Regression | 0.7933 | 0.7052 | 0.7248 | 0.7769 |

<br/>

### 5-Fold Stratified Cross-Validation

<br/>

| Model | CV AUC Mean | CV AUC Std | CV F1 | Stability |
|:------|:-----------:|:----------:|:-----:|:---------:|
| XGBoost (Tuned) | 0.8012 | ±0.0026 | 0.7200 | 🟢 Stable |
| LightGBM (Tuned) | 0.8013 | ±0.0026 | 0.7212 | 🟢 Stable |
| Logistic Regression | 0.7906 | ±0.0026 | 0.7081 | 🟢 Stable |

<br/>

> **📌 Literature Context:** The 2025 benchmark study on this exact dataset (medRxiv)
> found XGBoost and LightGBM as top performers at AUC ~0.80.
> **This model matches published state-of-the-art** on 68,634 real patient records.
> The moderate ceiling reflects genuine biological complexity — not a modeling failure.

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 3 — DATASET -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 📦 &nbsp; Dataset

</div>

**Source:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) — Kaggle (sulianova)

<div align="center">

<br/>

| Property | Value |
|:---------|:-----:|
| 📏 Raw size | 70,000 patients |
| ✅ After cleaning | 68,634 patients |
| 🎯 Target | Binary (CVD / No CVD) |
| ⚖️ Class balance | 49.5% CVD — 50.5% No CVD |
| 🧪 Train / Test split | 54,907 / 13,727 (80/20 stratified) |
| 🏥 Origin | Real clinical examination data |

<br/>

### Raw Features

<br/>

| Feature | Type | Description |
|:--------|:----:|:------------|
| `age` | Continuous | Age in days → converted to years |
| `gender` | Binary | 1 = Female, 2 = Male |
| `height` | Continuous | Height in cm |
| `weight` | Continuous | Weight in kg |
| `ap_hi` | Continuous | Systolic blood pressure (mmHg) |
| `ap_lo` | Continuous | Diastolic blood pressure (mmHg) |
| `cholesterol` | Ordinal | 1 = Normal, 2 = Above, 3 = Well above |
| `gluc` | Ordinal | 1 = Normal, 2 = Above, 3 = Well above |
| `smoke` | Binary | 0 = Non-smoker, 1 = Smoker |
| `alco` | Binary | 0 = No alcohol, 1 = Alcohol |
| `active` | Binary | 0 = Inactive, 1 = Active |

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 4 — FEATURE ENGINEERING -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 🔬 &nbsp; Clinical Feature Engineering

</div>

Nine new features were engineered from domain knowledge, grounded in published clinical guidelines. This is what separates a tutorial project from real clinical ML.

<div align="center"><br/>

```
RAW FEATURES  ──────────────────────────────────────────────────────►  ENGINEERED FEATURES
                                                                      
  ap_hi ──┐                                                           bp_stage      (JNC-8)
  ap_lo ──┤──► JNC-8 Hypertension Staging ─────────────────────────► 0 Normal
           │   Stage 0 → 4 encodes non-linear CVD risk                1 Elevated
           │                                                          2 Hypertension S1
           └──► Pulse Pressure = ap_hi - ap_lo ──────────────────────► pulse_pressure
                Mean Arterial Pressure = ap_lo + PP/3 ───────────────► map
                Pulse Pressure Category (≤40 / 41-60 / >60) ─────────► pp_category

  height ─┐
  weight ─┴──► BMI = weight / (height/100)² ──────────────────────────► bmi (capped at 60)
               WHO Classification (Underweight→Obese II+) ──────────────► bmi_category

  age ───────► Age in Years ───────────────────────────────────────────► age_years
               ESC Decade Groups (<40, 40-49, 50-59, 60+) ─────────────► age_group

  age_years + ap_hi ──► Standardised Interaction Term ────────────────► age_bp_interaction
  (Framingham: age × systolic BP is superadditive in CVD risk)

  cholesterol + gluc ──► Both ≥ above normal? ─────────────────────────► metabolic_risk
  smoke + alco + ¬active ──► Additive burden score ────────────────────► lifestyle_burden
  bp_stage≥2 AND bmi_cat≥3 ──► ACC/AHA high-risk cluster ─────────────► htn_obese
```

<br/>

### Feature Importance — SHAP (XGBoost, test set)

<br/>

| Rank | Feature | Mean \|SHAP\| | Clinical Meaning |
|:----:|:--------|:-------------:|:-----------------|
| 🔴 1 | `ap_hi` (Systolic BP) | Highest | Primary driver — hypertension is the #1 CVD risk factor |
| 🔴 2 | `ap_lo` (Diastolic BP) | High | Sustained pressure load on arterial walls |
| 🔴 3 | `pulse_pressure` | High | Marker of arterial stiffness (>60 mmHg = high risk) |
| 🟠 4 | `age_years` | Moderate | Cumulative vascular damage over decades |
| 🟠 5 | `cholesterol` | Moderate | Plaque formation and atherosclerosis |
| 🟠 6 | `bp_stage` | Moderate | Non-linear encoding of JNC-8 hypertension categories |
| 🟡 7 | `bmi` | Low-Moderate | Obesity-related cardiac strain |
| 🟡 8 | `map` | Low-Moderate | Average perfusion pressure |
| ⚪ 9 | `active`, `smoke`, `alco` | Near-zero | Self-reported lifestyle features — noisy signal |

> **🔍 Finding:** Self-reported lifestyle features (smoking, alcohol, activity) added
> **zero mutual information signal** beyond clinical measurements, confirming that
> objective vitals dominate CVD prediction in this dataset.

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 5 — METHODOLOGY FLOWCHART -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## ⚙️ &nbsp; Methodology

</div>

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FULL METHODOLOGY FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐
  │  RAW DATA        │  70,000 rows · 11 features · semicolon CSV
  │  (Kaggle)        │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  DATA CLEANING   │  Remove physiologically impossible BP, height, weight
  │                  │  Clinical thresholds (not arbitrary percentiles)
  │                  │  → 68,634 rows retained (1,366 removed = 1.95%)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  FEATURE ENG.    │  9 new clinical features (JNC-8, WHO BMI, MAP, PP,
  │                  │  age interaction, metabolic risk, lifestyle burden)
  │                  │  → 20 total features
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐      ┌─────────────────────────────────────────┐
  │  TRAIN/TEST      │      │  80% TRAIN (54,907)                     │
  │  SPLIT           │─────►│  → 5-Fold Stratified CV within train    │
  │  Stratified      │      │  20% TEST  (13,727) — touched ONCE      │
  └────────┬─────────┘      └─────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────┐
  │  PREPROCESSING   │  StandardScaler on continuous features only
  │  PIPELINE        │  Passthrough on binary/ordinal (already numeric)
  │                  │  Fit on train → transform both (no data leakage)
  └────────┬─────────┘
           │
           ├──────────────────────────┬──────────────────────────┐
           ▼                          ▼                          ▼
  ┌──────────────┐          ┌──────────────────┐       ┌──────────────────┐
  │  Logistic    │          │  XGBoost         │       │  LightGBM        │
  │  Regression  │          │  + Optuna 60     │       │  + Optuna 60     │
  │  C=0.05 L2   │          │  trials (CUDA)   │       │  trials (CPU)    │
  │  AUC: 0.7906 │          │  AUC: 0.8012     │       │  AUC: 0.8013     │
  └──────┬───────┘          └────────┬─────────┘       └────────┬─────────┘
         │                           │                           │
         └───────────────────────────┴───────────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  SOFT VOTING          │
                          │  ENSEMBLE             │
                          │  (average probas)     │
                          │  AUC: 0.8020          │
                          └──────────┬────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  ISOTONIC             │
                          │  CALIBRATION          │
                          │  Brier: 0.1812        │
                          └──────────┬────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  THRESHOLD            │
                          │  OPTIMISATION         │
                          │  Clinical: recall≥0.80│
                          │  Best F1: argmax F1   │
                          └──────────┬────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  SHAP                 │
                          │  EXPLAINABILITY       │
                          │  TreeExplainer        │
                          │  Beeswarm · Waterfall │
                          │  Dependence plots     │
                          └──────────┬────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  🚀 GRADIO DEPLOY     │
                          │  Kaggle → share=True  │
                          │  Public gradio.live   │
                          └──────────────────────┘
```

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 6 — EXPLAINABILITY -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 🧠 &nbsp; Explainability — Why SHAP?

</div>

Most ML projects stop at accuracy. This project explains **every single prediction** using SHAP (SHapley Additive exPlanations) — the gold standard for model interpretability in clinical AI.

<div align="center"><br/>

| Plot | What it shows | Why it matters |
|:-----|:-------------|:---------------|
| 🐝 **Beeswarm** | Every feature × every patient impact | Global view — which features drive the model across all patients |
| 📊 **Bar Chart** | Mean \|SHAP\| per feature | Ranked, model-honest importance (better than XGBoost's built-in gain) |
| 🌊 **Waterfall** | Feature contributions for one patient | Local view — *why* the model flagged **this specific patient** |
| 🔵 **Dependence** | Feature value vs SHAP value, coloured by interaction | Shows the exact BP threshold where risk jumps — matches JNC-8 Stage 2 |

<br/>

```
SHAP WATERFALL — Example: High-Risk Patient (P = 0.81)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Base value (avg prediction)          │  0.50
                                       │
  ap_hi = 165 mmHg         ████████████│+0.18  ← Stage 2 hypertension
  age_years = 61            ██████     │+0.09  ← Older age compounds BP
  bp_stage = 3              █████      │+0.08  ← Explicit JNC-8 encoding
  cholesterol = 3           ████       │+0.06  ← Well above normal
  pulse_pressure = 75 mmHg  ███        │+0.04  ← Wide PP → arterial stiffness
  bmi = 31.2                ██         │+0.03  ← Obese class I
  active = 0                █          │+0.01  ← Inactive
  smoke = 1                 ▒          │+0.00  ← Near-zero signal
  ─────────────────────────────────────┼──────
  FINAL PREDICTION                     │  0.81  → 🔴 HIGH RISK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 7 — CALIBRATION & THRESHOLDS -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 📐 &nbsp; Calibration & Threshold Optimisation

</div>

Raw model probabilities are not inherently trustworthy. A model saying "70% chance of CVD" should mean exactly 70 out of 100 such patients actually have CVD. This project applies **isotonic regression calibration** to achieve this property.

<div align="center"><br/>

```
CALIBRATION RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Raw Ensemble                 Calibrated Ensemble
  ─────────────────            ──────────────────────
  Brier Score: ~0.186          Brier Score: 0.1812 ✅
  Slightly over-confident      Well-calibrated curve

  When model says 0.7 →        When model says 0.7 →
  ~67% are actually CVD        ~70% are actually CVD ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

<br/>

### Threshold Analysis

<br/>

| Threshold | Sensitivity (Recall) | Specificity | F1 Score | Use Case |
|:---------:|:--------------------:|:-----------:|:--------:|:---------|
| 0.50 | 0.69 | 0.77 | 0.72 | Default (equal FP/FN cost) |
| **Clinical** | **≥ 0.80** | Lower | Higher recall | **Deployed** — missing CVD is worse than false alarm |
| Best F1 | Balanced | Balanced | Maximum | Research benchmarking |

<br/>

> **Clinical reasoning:** In cardiovascular screening, a **false negative** (missing a
> CVD patient) is far more dangerous than a **false positive** (unnecessary follow-up).
> The deployed app uses the clinical threshold to prioritise sensitivity.

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 8 — NOTEBOOKS / STRUCTURE -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 📁 &nbsp; Repository Structure

</div>

```
cardiorisk-ai/
│
├── 📓 cardiorisk_classifier.ipynb   ← Main Kaggle notebook (all 5 snippets)
│
├── 🤖 models/
│   ├── xgb_model.pkl                ← Tuned XGBoost (Optuna, 60 trials)
│   ├── lgbm_model.pkl               ← Tuned LightGBM (Optuna, 60 trials)
│   ├── preprocessor.pkl             ← ColumnTransformer (StandardScaler)
│   └── calibrator.pkl               ← Isotonic regression calibrator
│
├── ⚙️ model_config.json             ← Feature lists + threshold values
│
├── 📊 outputs/
│   ├── eda_distributions.png        ← Snippet 1: feature distributions
│   ├── eda_categorical.png          ← Snippet 1: categorical CVD rates
│   ├── eda_correlation.png          ← Snippet 1: Pearson heatmap
│   ├── eda_scatter.png              ← Snippet 1: age vs BP scatter
│   ├── feature_importance_mi.png    ← Snippet 2: mutual information
│   ├── feature_importance_perm.png  ← Snippet 2: permutation importance
│   ├── cv_comparison.png            ← Snippet 3: CV metric bars
│   ├── roc_pr_curves.png            ← Snippet 3: ROC + PR curves
│   ├── cv_fold_stability.png        ← Snippet 3: AUC per fold
│   ├── roc_before_after_fe.png      ← Snippet 3C: FE ablation study
│   ├── shap_beeswarm.png            ← Snippet 4: global SHAP
│   ├── shap_bar.png                 ← Snippet 4: mean |SHAP|
│   ├── shap_waterfall.png           ← Snippet 4: patient-level SHAP
│   ├── shap_dependence.png          ← Snippet 4: dependence plots
│   ├── calibration.png              ← Snippet 4: calibration curves
│   ├── threshold_analysis.png       ← Snippet 4: precision/recall/F1
│   └── confusion_matrices.png       ← Snippet 4: default vs clinical
│
└── 📖 README.md                     ← This file
```

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 9 — NOTEBOOK SNIPPETS -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 📓 &nbsp; Notebook Structure — 5 Snippets

</div>

<div align="center">

| Snippet | Title | Key Outputs |
|:-------:|:------|:------------|
| **1** | 🔍 Setup + EDA | 4 EDA plots · outlier removal · class balance |
| **2** | 🧹 Preprocessing + Feature Selection | MI scores · permutation importance · CV splitter |
| **3** | 🤖 Model Training + CV | 5-fold CV · ROC/PR curves · fold stability · Optuna tuning |
| **3C** | 🔬 Clinical Feature Engineering | 9 new features · ablation study · FE vs no-FE comparison |
| **4** | 📊 Explainability + Evaluation | SHAP (4 plots) · calibration · thresholds · confusion matrices |
| **5** | 🚀 Gradio Deployment | Live Kaggle-hosted app · SHAP per patient · risk tier cards |

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 10 — TECH STACK -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 🛠️ &nbsp; Tech Stack

<br/>

| Layer | Technology | Purpose |
|:------|:----------:|:--------|
| **Language** | ![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white) | Core language |
| **ML — Boosting** | ![XGBoost](https://img.shields.io/badge/XGBoost-E34234?style=flat-square) ![LightGBM](https://img.shields.io/badge/LightGBM-9AC122?style=flat-square) | Primary classifiers |
| **ML — Baseline** | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | Logistic Regression · pipelines · CV · calibration |
| **Hyperparameter Tuning** | ![Optuna](https://img.shields.io/badge/Optuna-3B4EFF?style=flat-square) | 60-trial TPE search per model |
| **Explainability** | ![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=flat-square) | TreeExplainer · beeswarm · waterfall · dependence |
| **Visualisation** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) ![Seaborn](https://img.shields.io/badge/Seaborn-4EACD4?style=flat-square) | All 17 output plots |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Preprocessing · feature engineering |
| **Deployment** | ![Gradio](https://img.shields.io/badge/Gradio_5.29-FF7C00?style=flat-square) | Interactive web app |
| **Compute** | ![Kaggle](https://img.shields.io/badge/Kaggle_GPU_(T4)-20BEFF?style=flat-square&logo=kaggle&logoColor=white) | XGBoost CUDA acceleration |

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 11 — KEY FINDINGS -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## 💡 &nbsp; Key Findings

</div>

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔑  FINDINGS THAT MAKE THIS PROJECT SCIENTIFICALLY INTERESTING             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. LIFESTYLE FEATURES ADD ZERO SIGNAL                                      │
│     Smoking, alcohol, and physical activity scored 0.000 mutual             │
│     information against the target. Permutation importance confirmed        │
│     all three near-zero. Objective clinical measurements dominate.          │
│                                                                             │
│  2. GRADIENT BOOSTING RE-DISCOVERS CLINICAL BOUNDARIES                      │
│     Explicit JNC-8 staging and WHO BMI categorisation produced no           │
│     improvement over raw features (+0.000 AUC). Tree models learn           │
│     these thresholds internally from data — matching published theory.      │
│                                                                             │
│  3. AUC 0.802 IS THE PUBLISHED CEILING, NOT A FAILURE                       │
│     The 2025 medRxiv benchmark on this exact 70k dataset found              │
│     XGBoost/LightGBM top at ~AUC 0.80. We match state-of-the-art.          │
│     Models reporting 0.95+ on this dataset are overfitting or leaking.      │
│                                                                             │
│  4. SYSTOLIC BP IS THE DOMINANT PREDICTOR                                   │
│     ap_hi alone accounts for more SHAP value than all lifestyle             │
│     features combined. This aligns with decades of cardiology literature.   │
│                                                                             │
│  5. CALIBRATION MATTERS MORE THAN ACCURACY                                  │
│     Isotonic calibration ensures that P=0.70 means ~70% of those            │
│     patients actually have CVD — essential for clinical trust.              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

</div>

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 12 — QUICK START -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## ⚡ &nbsp; Quick Start

</div>

**Option 1 — Try the live demo (no setup)**

Click the **Live Demo** button at the top of this README.

**Option 2 — Run on Kaggle (recommended)**

1. Open the [Kaggle Notebook](#) and click **Copy & Edit**
2. Add the dataset: `Notebook Settings → Add Data → "cardiovascular-disease-dataset"` by sulianova
3. Enable GPU: `Notebook Settings → Accelerator → GPU T4 x2`
4. Run all cells in order (Snippets 1 → 5)
5. The Gradio `share=True` link appears at the end of Snippet 5

**Option 3 — Run locally**

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/cardiorisk-ai.git
cd cardiorisk-ai

# Install dependencies
pip install xgboost lightgbm scikit-learn shap gradio==5.29.0 \
            optuna pandas numpy matplotlib seaborn joblib

# Download dataset from Kaggle and place cardio_train.csv in /data/
# Then run the notebook top-to-bottom
jupyter notebook cardiorisk_classifier.ipynb
```

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- SECTION 13 — LIMITATIONS -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

## ⚠️ &nbsp; Limitations & Ethical Considerations

</div>

| Limitation | Detail |
|:-----------|:-------|
| 📸 **Snapshot data** | BP and cholesterol measured once — CVD develops over years of sustained exposure |
| 🗂️ **No ECG / imaging** | Datasets with ST-segment, ejection fraction achieve 0.92+ AUC — richer features exist |
| 🌍 **Population bias** | Dataset origin is not fully documented — may not generalise to all ethnicities |
| 🔢 **Self-reported lifestyle** | Smoking and alcohol are self-reported — notoriously unreliable in clinical surveys |
| 🏥 **Not for clinical use** | This is an educational portfolio project. Any clinical deployment requires regulatory approval, prospective validation, and physician oversight |

<br/>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- FOOTER -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div align="center">

<br/>

---

**Built with 🫀 for a Computer Science portfolio**

*Clean code · Honest metrics · Clinical grounding · Full deployment*

<br/>

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f?style=flat-square&logo=python)](https://python.org)
[![Trained on Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle-20BEFF?style=flat-square&logo=kaggle)](https://kaggle.com)
[![Explained with SHAP](https://img.shields.io/badge/Explained%20with-SHAP-FF6B6B?style=flat-square)](https://shap.readthedocs.io)
[![Deployed with Gradio](https://img.shields.io/badge/Deployed%20with-Gradio-FF7C00?style=flat-square)](https://gradio.app)

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
