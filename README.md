<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=e74c3c,c0392b,ff6b9d,ff4081&height=220&section=header&text=🫀%20CardioRisk%20AI&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Clinical-Grade%20Cardiovascular%20Disease%20Classifier&descAlignY=60&descSize=18&descColor=ffb3b3" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=4FC3F7&center=true&vCenter=true&width=700&lines=Predict+Cardiovascular+Disease+Risk;Explain+Every+Prediction+with+SHAP;Calibrated+%7C+Cross-Validated+%7C+Deployed;AUC+0.802+%E2%80%94+State-of-the-Art+on+68%2C634+Patients)](https://git.io/typing-svg)

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Tuned-E34234?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-Tuned-9AC122?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Gradio-5.29.0-FF7C00?style=for-the-badge&logo=gradio&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/ROC--AUC-0.802-2ECC71?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-68%2C634%20Patients-3498DB?style=flat-square&logo=kaggle"/>
  <img src="https://img.shields.io/badge/Validation-5--Fold%20Stratified%20CV-9B59B6?style=flat-square"/>
  <img src="https://img.shields.io/badge/Calibration-Isotonic%20Regression-E67E22?style=flat-square"/>
  <img src="https://img.shields.io/badge/Optuna-60%20Trials%20Each-3B4EFF?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-95A5A6?style=flat-square"/>
</p>

<br/>

<a href="#">
  <img src="https://img.shields.io/badge/🚀%20Live%20Demo%20→%20Try%20it%20Now-E74C3C?style=for-the-badge" height="40"/>
</a>
&nbsp;
<a href="#">
  <img src="https://img.shields.io/badge/📓%20Kaggle%20Notebook%20→%20Open-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" height="40"/>
</a>

<br/><br/>

> ### *"A machine learning system that predicts cardiovascular disease risk*
> ### *and explains every prediction — the way a clinician would."*

<br/>

</div>

---

## 📋 Overview

**CardioRisk AI** is a full-stack, production-grade machine learning project that classifies cardiovascular disease (CVD) risk from routine clinical measurements. It demonstrates the **complete ML lifecycle** — from raw data ingestion and clinical feature engineering to model explainability, probability calibration, and interactive deployment.

Built on **68,634 real patient records**, the system achieves **ROC-AUC 0.802**, matching the published state-of-the-art benchmark on this dataset (medRxiv 2025). Every design decision is clinically grounded and statistically justified.

<br/>

<div align="center">

### ✨ What Makes This Different

| | This Project | Typical Tutorial |
|:-|:------------|:----------------|
| **Metrics** | AUC · F1 · Calibration · Brier Score | Accuracy only |
| **Validation** | Stratified 5-Fold CV | Single train/test split |
| **Explainability** | SHAP beeswarm · waterfall · dependence | None |
| **Calibration** | Isotonic regression | Raw probabilities |
| **Threshold** | Clinical sensitivity-optimised | Default 0.5 |
| **Features** | 9 clinical features engineered | Raw features only |
| **Tuning** | Optuna 60 trials per model | Default params |
| **Deployment** | Live Gradio app | None |

</div>

<br/>

---

## 🏗️ System Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#1a1a2e',
  'primaryTextColor': '#ffffff',
  'primaryBorderColor': '#4FC3F7',
  'lineColor': '#4FC3F7',
  'secondaryColor': '#16213e',
  'tertiaryColor': '#0f3460',
  'edgeLabelBackground': '#1a1a2e',
  'clusterBkg': '#16213e',
  'titleColor': '#4FC3F7'
}}}%%

flowchart TD
    A([🗂️ Raw Dataset\n70,000 patients\n11 features]) --> B

    subgraph CLEAN ["🧹 Data Cleaning"]
        B[Outlier Removal\nClinical BP thresholds\n60-250 systolic\n40-150 diastolic]
        B --> C[68,634 clean rows\n1.95% removed]
    end

    C --> D

    subgraph ENGINEER ["🔬 Clinical Feature Engineering"]
        D[JNC-8 BP Staging\nStage 0 → 4]
        D --> E[WHO BMI Classification\nUnderweight → Obese II+]
        E --> F[MAP · Pulse Pressure\nAge × BP Interaction\nMetabolic Risk Flag]
        F --> G[20 total features\nfrom 11 raw inputs]
    end

    G --> H

    subgraph SPLIT ["📊 Data Split"]
        H[Stratified 80/20 Split]
        H --> I[Train\n54,907 patients]
        H --> J[Test\n13,727 patients\nTouched ONCE]
    end

    I --> K

    subgraph PREPROCESS ["⚙️ Preprocessing Pipeline"]
        K[ColumnTransformer\nStandardScaler → Continuous\nPassthrough → Binary & Ordinal\nFit on Train Only]
    end

    K --> L
    K --> M
    K --> N

    subgraph MODELS ["🤖 Model Training + Optuna Tuning"]
        L[Logistic Regression\nC=0.05 L2\nAUC: 0.7906]
        M[XGBoost\n60-trial TPE Search\nCUDA Accelerated\nAUC: 0.8012]
        N[LightGBM\n60-trial TPE Search\nAll CPU Cores\nAUC: 0.8013]
    end

    L --> O
    M --> O
    N --> O

    subgraph ENSEMBLE ["🎯 Ensemble + Calibration"]
        O[Soft Voting Ensemble\nAverage Probabilities\nAUC: 0.8020]
        O --> P[Isotonic Calibration\nBrier Score: 0.1812]
        P --> Q[Threshold Optimisation\nClinical: Recall ≥ 0.80\nBest F1: argmax F1]
    end

    Q --> R
    Q --> S

    subgraph OUTPUT ["📈 Outputs"]
        R[SHAP Explainability\nBeeswarm · Waterfall\nDependence · Bar]
        S[🚀 Gradio App\nKaggle-Hosted\nPublic URL]
    end

    style CLEAN fill:#0f3460,stroke:#4FC3F7,color:#fff
    style ENGINEER fill:#1a1a2e,stroke:#E74C3C,color:#fff
    style SPLIT fill:#16213e,stroke:#2ECC71,color:#fff
    style PREPROCESS fill:#0f3460,stroke:#9B59B6,color:#fff
    style MODELS fill:#1a1a2e,stroke:#F39C12,color:#fff
    style ENSEMBLE fill:#16213e,stroke:#4FC3F7,color:#fff
    style OUTPUT fill:#0f3460,stroke:#2ECC71,color:#fff
```

<br/>

---

## 🔄 Pipeline & Data Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#1a1a2e',
  'primaryTextColor': '#ffffff',
  'primaryBorderColor': '#7C6AF7',
  'lineColor': '#7C6AF7',
  'secondaryColor': '#16213e',
  'actorBkg': '#0f3460',
  'actorTextColor': '#ffffff',
  'actorBorderColor': '#4FC3F7',
  'activationBkgColor': '#16213e',
  'activationBorderColor': '#4FC3F7',
  'noteBkgColor': '#0f3460',
  'noteTextColor': '#ffffff',
  'noteBorderColor': '#7C6AF7',
  'signalColor': '#4FC3F7',
  'signalTextColor': '#ffffff',
  'labelBoxBkgColor': '#1a1a2e',
  'labelBoxBorderColor': '#7C6AF7',
  'labelTextColor': '#ffffff',
  'loopTextColor': '#ffffff',
  'sequenceNumberColor': '#ffffff'
}}}%%

sequenceDiagram
    participant U as 👤 User / Clinician
    participant G as 🖥️ Gradio App
    participant FE as 🔬 Feature Engine
    participant PP as ⚙️ Preprocessor
    participant XGB as 🤖 XGBoost
    participant LGBM as 🤖 LightGBM
    participant CAL as 📐 Calibrator
    participant SH as 🧠 SHAP Explainer

    U->>G: Enter vitals (age, BP, weight, cholesterol...)
    G->>FE: Raw 11 features

    Note over FE: JNC-8 BP Staging<br/>WHO BMI Category<br/>MAP · Pulse Pressure<br/>Age × BP Interaction

    FE->>PP: 20 engineered features
    PP->>PP: StandardScaler on continuous<br/>Passthrough binary/ordinal

    PP->>XGB: Scaled feature vector
    PP->>LGBM: Scaled feature vector

    XGB-->>CAL: P(CVD) = 0.73
    LGBM-->>CAL: P(CVD) = 0.76

    Note over CAL: Soft average → 0.745<br/>Isotonic calibration<br/>→ calibrated P(CVD)

    CAL-->>G: Calibrated probability

    G->>SH: Feature vector for explanation
    SH->>SH: TreeExplainer on XGBoost<br/>Compute exact Shapley values

    SH-->>G: SHAP values per feature

    Note over G: Apply clinical threshold<br/>Recall ≥ 0.80 sensitivity target

    G-->>U: 🔴 Risk tier + probability
    G-->>U: 📊 SHAP waterfall plot
    G-->>U: 📋 Computed vitals card
```

<br/>

---

## 🔬 Clinical Feature Engineering

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#0f3460',
  'primaryTextColor': '#ffffff',
  'primaryBorderColor': '#E74C3C',
  'lineColor': '#E74C3C',
  'secondaryColor': '#16213e',
  'tertiaryColor': '#1a1a2e',
  'edgeLabelBackground': '#16213e',
  'clusterBkg': '#1a1a2e'
}}}%%

flowchart LR
    subgraph RAW ["📥 Raw Features"]
        R1(ap_hi\nSystolic BP)
        R2(ap_lo\nDiastolic BP)
        R3(height\nweight)
        R4(age\ndays)
        R5(cholesterol\ngluc)
        R6(smoke\nalco\nactive)
    end

    subgraph DERIVED ["⚗️ Derived Features"]
        D1["🩺 bp_stage\nJNC-8 Stage 0→4\nNon-linear CVD risk\nencoding"]
        D2["💓 pulse_pressure\nap_hi − ap_lo\nArterial stiffness\nmarker"]
        D3["🔬 map\nMean Arterial Pressure\nap_lo + PP/3\nPerfusion load"]
        D4["📏 bmi\nWeight÷Height²\ncapped at 60\nbmi_category WHO"]
        D5["🗓️ age_years\nDays → Years\nage_group ESC\ndecades"]
        D6["⚡ age_bp_interaction\nStandardised\nage × systolic\nFramingham term"]
        D7["⚠️ metabolic_risk\nChol≥2 AND Gluc≥2\nMetabolic syndrome\nflag"]
        D8["🏃 lifestyle_burden\nSmoke+Alco+¬Active\nAdditive score\n0 → 3"]
        D9["🚨 htn_obese\nBP Stage≥2 AND\nBMI Cat≥3\nACC/AHA cluster"]
    end

    R1 --> D1
    R2 --> D1
    R1 --> D2
    R2 --> D2
    R2 --> D3
    D2 --> D3
    R3 --> D4
    R4 --> D5
    D5 --> D6
    R1 --> D6
    R5 --> D7
    R6 --> D8
    D1 --> D9
    D4 --> D9

    style RAW fill:#1a1a2e,stroke:#4FC3F7,color:#fff
    style DERIVED fill:#16213e,stroke:#E74C3C,color:#fff
```

<br/>

---

## 🏆 Results & Metrics

<div align="center">

### Held-Out Test Set Performance — 13,727 Patients

| Rank | Model | ROC-AUC | F1 Score | Accuracy | Avg Precision |
|:----:|:------|:-------:|:--------:|:--------:|:-------------:|
| 🥇 | **LightGBM (Tuned)** | **0.8021** | **0.7211** | **0.7341** | **0.7843** |
| 🥈 | Soft Ensemble (5 models) | 0.8020 | 0.7184 | 0.7326 | 0.7844 |
| 🥉 | XGBoost (Tuned) | 0.8017 | 0.7193 | 0.7333 | 0.7834 |
| — | XGBoost (Default) | 0.8003 | 0.7189 | 0.7327 | 0.7816 |
| — | LightGBM (Default) | 0.7985 | 0.7186 | 0.7321 | 0.7800 |
| — | Logistic Regression | 0.7933 | 0.7052 | 0.7248 | 0.7769 |

<br/>

### 5-Fold Stratified Cross-Validation

| Model | CV AUC | Std | CV F1 | CV Accuracy | Stability |
|:------|:------:|:---:|:-----:|:-----------:|:---------:|
| XGBoost (Tuned) | 0.8012 | ±0.0026 | 0.7200 | 0.7355 | 🟢 |
| LightGBM (Tuned) | 0.8013 | ±0.0026 | 0.7212 | 0.7354 | 🟢 |
| Logistic Regression | 0.7906 | ±0.0026 | 0.7081 | 0.7281 | 🟢 |

<br/>

### Calibration

| Metric | Value | Interpretation |
|:-------|:-----:|:--------------|
| Brier Score | **0.1812** | Mean squared probability error (lower = better) |
| Calibration Method | Isotonic Regression | Non-parametric, data-driven |
| When model says 70% | ~70% actually have CVD | ✅ Well-calibrated |

<br/>

> **📌 Literature Context:** The 2025 medRxiv benchmark on this exact Kaggle 70k dataset found
> XGBoost and LightGBM as top performers at AUC ~0.80.
> **This model matches published state-of-the-art.** Models claiming 0.95+ on this
> dataset are overfitting or leaking data.

</div>

<br/>

---

## 🧠 SHAP Explainability

<div align="center">

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#1a1a2e',
  'primaryTextColor': '#ffffff',
  'primaryBorderColor': '#FF6B6B',
  'lineColor': '#FF6B6B',
  'secondaryColor': '#16213e',
  'clusterBkg': '#0f3460'
}}}%%

flowchart TD
    M([🤖 XGBoost Model\nTreeExplainer]) --> A & B & C & D

    subgraph PLOTS ["📊 SHAP Plot Suite"]
        A["🐝 Beeswarm Plot\nAll patients × All features\nGlobal importance view\nColour = feature value"]
        B["📊 Bar Chart\nMean |SHAP| per feature\nModel-honest ranking\nBeats built-in gain importance"]
        C["🌊 Waterfall Plot\nOne patient at a time\nWhy THIS prediction?\nFeature-level breakdown"]
        D["🔵 Dependence Plot\nFeature value vs SHAP\nInteraction colouring\nShows clinical thresholds"]
    end

    A --> E([📋 Global Understanding\nWhich features matter\nacross all patients])
    B --> E
    C --> F([👤 Local Understanding\nWhy was THIS patient\nflagged as high risk])
    D --> G([📈 Threshold Discovery\nModel learns JNC-8\nStage 2 boundary\nautomatically])

    style PLOTS fill:#16213e,stroke:#FF6B6B,color:#fff
```

</div>

<div align="center">

### Feature Importance — SHAP Rankings

| Rank | Feature | Importance | Clinical Meaning |
|:----:|:--------|:----------:|:-----------------|
| 🔴 1 | `ap_hi` — Systolic BP | ████████████ Highest | Primary CVD driver — hypertension #1 risk factor |
| 🔴 2 | `ap_lo` — Diastolic BP | ██████████ High | Sustained arterial wall pressure |
| 🔴 3 | `pulse_pressure` | █████████ High | Arterial stiffness marker (>60 mmHg = high risk) |
| 🟠 4 | `age_years` | ███████ Moderate | Cumulative vascular damage |
| 🟠 5 | `cholesterol` | ██████ Moderate | Plaque formation, atherosclerosis |
| 🟠 6 | `bp_stage` (JNC-8) | █████ Moderate | Non-linear hypertension encoding |
| 🟡 7 | `bmi` | ████ Low-Mod | Obesity-related cardiac load |
| 🟡 8 | `map` | ███ Low-Mod | Average perfusion pressure |
| ⚪ 9 | `smoke` / `alco` / `active` | ▒ Near-zero | Self-reported, noisy signal |

> **🔍 Key Finding:** Lifestyle features (smoke, alco, active) scored **mutual information = 0.000**
> against the target. Tree models confirm this — objective clinical measurements dominate CVD prediction.

</div>

<br/>

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Role |
|:------|:----------:|:-----|
| **Language** | ![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white) | Core runtime |
| **Primary Models** | ![XGBoost](https://img.shields.io/badge/XGBoost-E34234?style=flat-square) ![LightGBM](https://img.shields.io/badge/LightGBM-9AC122?style=flat-square) | Gradient boosted classifiers |
| **Baseline + Pipelines** | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | LR · CV · calibration · metrics |
| **Hyperparameter Tuning** | ![Optuna](https://img.shields.io/badge/Optuna_TPE-3B4EFF?style=flat-square) | 60-trial Bayesian search per model |
| **Explainability** | ![SHAP](https://img.shields.io/badge/SHAP_TreeExplainer-FF6B6B?style=flat-square) | Exact Shapley values |
| **Visualisation** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) ![Seaborn](https://img.shields.io/badge/Seaborn-4EACD4?style=flat-square) | 17 output plots |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Preprocessing + engineering |
| **Deployment** | ![Gradio](https://img.shields.io/badge/Gradio_5.29-FF7C00?style=flat-square) | Interactive web app |
| **Compute** | ![Kaggle](https://img.shields.io/badge/Kaggle_GPU_T4-20BEFF?style=flat-square&logo=kaggle&logoColor=white) | CUDA-accelerated XGBoost |

</div>

<br/>

---

## 📁 Project Structure

```
cardiorisk-ai/
│
├── 📓 cardiorisk_classifier.ipynb     ← Main Kaggle notebook (5 snippets)
│
├── 🤖 models/
│   ├── xgb_model.pkl                  ← Tuned XGBoost (Optuna, 60 trials, CUDA)
│   ├── lgbm_model.pkl                 ← Tuned LightGBM (Optuna, 60 trials, CPU)
│   ├── preprocessor.pkl               ← ColumnTransformer (StandardScaler)
│   └── calibrator.pkl                 ← Isotonic regression calibrator
│
├── ⚙️  model_config.json              ← Feature lists + clinical threshold values
│
├── 📊 outputs/
│   ├── eda_distributions.png          ← Feature distributions by CVD class
│   ├── eda_categorical.png            ← Categorical CVD prevalence rates
│   ├── eda_correlation.png            ← Pearson correlation heatmap
│   ├── eda_scatter.png                ← Age vs BP coloured by target
│   ├── feature_importance_mi.png      ← Mutual information scores
│   ├── feature_importance_perm.png    ← Permutation importance (LR)
│   ├── cv_comparison.png              ← 5-fold CV metric comparison
│   ├── roc_pr_curves.png              ← ROC + Precision-Recall curves
│   ├── cv_fold_stability.png          ← AUC per fold (stability view)
│   ├── roc_before_after_fe.png        ← Feature engineering ablation
│   ├── shap_beeswarm.png              ← Global SHAP (all patients)
│   ├── shap_bar.png                   ← Mean |SHAP| feature ranking
│   ├── shap_waterfall.png             ← High-risk vs low-risk patient
│   ├── shap_dependence.png            ← ap_hi and age_years dependence
│   ├── calibration.png                ← Calibration curves + Brier scores
│   ├── threshold_analysis.png         ← Precision/Recall/F1 vs threshold
│   └── confusion_matrices.png         ← Default vs clinical threshold
│
└── 📖 README.md
```

<br/>

---

## ⚡ Installation & Quick Start

<div align="center">

### Option 1 — Live Demo (No Setup)

<a href="#">
  <img src="https://img.shields.io/badge/🚀%20Open%20Live%20Demo-E74C3C?style=for-the-badge" height="38"/>
</a>

<br/><br/>

### Option 2 — Kaggle (Recommended)

</div>

1. Open the [Kaggle Notebook](#) and click **Copy & Edit**
2. Add dataset: `Notebook Settings → Add Data → cardiovascular-disease-dataset` (sulianova)
3. Enable GPU: `Notebook Settings → Accelerator → GPU T4 x2`
4. Run all cells top to bottom (Snippets 1 → 5)
5. Public Gradio URL appears at end of Snippet 5

<div align="center">

### Option 3 — Local

</div>

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/cardiorisk-ai.git
cd cardiorisk-ai

# Install
pip install xgboost lightgbm scikit-learn shap gradio==5.29.0 \
            optuna pandas numpy matplotlib seaborn joblib

# Place cardio_train.csv from Kaggle into /data/
# Run notebook
jupyter notebook cardiorisk_classifier.ipynb
```

<br/>

---

## 💡 Key Findings

<div align="center">

| # | Finding | Implication |
|:-:|:--------|:------------|
| **1** | 🚭 Lifestyle features (smoke, alco, active) scored **MI = 0.000** | Self-reported lifestyle data adds no predictive signal beyond objective vitals |
| **2** | 🌳 Clinical feature engineering (JNC-8, WHO BMI) gained **+0.000 AUC** | Gradient boosting autonomously discovers clinical thresholds from raw data |
| **3** | 📊 AUC 0.802 matches the **2025 medRxiv benchmark ceiling** | This is not a failure — it reflects genuine biological noise in snapshot measurements |
| **4** | 💉 `ap_hi` SHAP dominates **all other features combined** | Systolic BP is the #1 CVD driver — consistent with 50 years of cardiology literature |
| **5** | 📐 Isotonic calibration improved **probability trustworthiness** | P=0.70 now means 70% of patients actually have CVD — essential for clinical deployment |

</div>

<br/>

---

## 🔮 Future Work

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#1a1a2e',
  'primaryTextColor': '#ffffff',
  'primaryBorderColor': '#2ECC71',
  'lineColor': '#2ECC71',
  'secondaryColor': '#16213e',
  'clusterBkg': '#0f3460'
}}}%%

flowchart LR
    NOW([📍 Current\nAUC 0.802\nGradio App\nSHAP]) --> A & B & C & D

    A["🧬 Richer Data\nECG features\nST-segment\nEjection fraction\n→ Target AUC 0.92+"]

    B["🏥 Multi-Dataset\nMerge Cleveland\nHungarian\nStatlog sources"]

    C["🤗 HuggingFace\nPermanent Spaces\ndeployment\nno 72h expiry"]

    D["📱 Clinical API\nREST endpoint\nFHIR-compatible\nEHR integration"]

    style NOW fill:#0f3460,stroke:#2ECC71,color:#fff
```

<br/>

---

## ⚠️ Limitations & Ethics

<div align="center">

| Limitation | Detail |
|:-----------|:-------|
| 📸 **Snapshot data** | BP/cholesterol measured once — CVD develops over years |
| 🗂️ **No ECG/imaging** | Richer features (ST-slope, ejection fraction) yield 0.92+ AUC |
| 🌍 **Population bias** | Dataset origin undocumented — may not generalise to all groups |
| 🔢 **Self-reported lifestyle** | Smoking and alcohol are notoriously unreliable in surveys |
| 🏥 **Not for clinical use** | Requires regulatory approval, prospective validation, and physician oversight |

</div>

<br/>

---

<div align="center">

**Built with 🫀 for a Computer Science portfolio**

*Clean code · Honest metrics · Clinical grounding · Full deployment*

<br/>

[![Python](https://img.shields.io/badge/Made%20with-Python-1f425f?style=flat-square&logo=python)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle-20BEFF?style=flat-square&logo=kaggle)](https://kaggle.com)
[![SHAP](https://img.shields.io/badge/Explained%20with-SHAP-FF6B6B?style=flat-square)](https://shap.readthedocs.io)
[![Gradio](https://img.shields.io/badge/Deployed%20with-Gradio-FF7C00?style=flat-square)](https://gradio.app)

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=e74c3c,c0392b,ff6b9d,ff4081&height=120&section=footer" width="100%"/>

</div>
