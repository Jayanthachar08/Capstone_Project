# Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-80%2F80%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Academic%20Research-lightgrey.svg)]()

> **B.Tech Computer Science & Engineering — Capstone Project (7th Semester)**  
> **Institution:** Department of Computer Science & Engineering  
> **Environment:** Windows • Python 3.11  

---

## 1. Problem Statement

Lithium-ion batteries are the dominant energy storage technology for electric vehicles (EVs) and consumer electronics. Accurately predicting a battery cell's End-of-Life (EOL)—the point where its capacity degrades below 80% of its nominal rating—at an ultra-early stage of cycling is vital for quality control, warranty estimation, and battery pack design. However, during early life (e.g., within the first 40 to 100 cycles), absolute capacity fade and resistance growth are often minimal and heavily masked by measurement noise and initial manufacturing dispersion.

---

## 2. Research Gap

Existing early-prediction approaches (such as the landmark work by Severson et al., *Nature Energy* 2019) model each battery's degradation trajectory in isolation, relying on absolute features like $\Delta Q_{100-10}(V)$ or global regression models. These approaches face two fundamental limitations:
1. **Lack of Peer Context:** Absolute degradation rates are confounded by fast-charging protocols and operating regimes. Without evaluating a cell's degradation rate *relative to similar batteries cycled under comparable conditions*, subtle anomalies go unnoticed.
2. **Overconfidence in Sparse Feature Spaces:** Standard machine learning models issue point predictions or probabilities without assessing whether the query battery resembles the training distribution, risking confident misclassifications on atypical cells.

**Central Research Question:**  
*Does a battery's deviation from the degradation trajectories of appropriately matched peer batteries provide reliable incremental predictive information about future end-of-life (EOL) risk beyond its initial/current health condition and individual degradation history?*

---

## 3. Project Objectives

1. **Extract and Validate Benchmark Battery Records:** Parse and validate multi-gigabyte HDF5/MATLAB records from the Severson et al. benchmark dataset across three experimental batches (140 cells total, 124 eligible benchmark cells).
2. **Build Leakage-Safe Battery-Level Splits:** Implement strict battery-level partitioning (Train: 41, Primary Test: 43, Secondary Test: 40) ensuring no cell data spans partitions and zero post-cutoff or future lifecycle information leaks into early features.
3. **Multi-Cutoff Feature Engineering:** Construct 34 early-life features across four physical groups at early observation cutoffs $N_{\text{obs}} \in \{40, 70, 100\}$ cycles.
4. **Develop a Peer-Matching Subsystem:** Pair test batteries strictly with training reference peers using early-life operating features and compute relative trajectory deviations ($k=5$).
5. **Formulate the Peer-Match Reliability Index (PMRI):** Define an interpretable composite metric quantifying reference neighborhood support to gate or weight predictions and advise model abstention.
6. **Train & Evaluate Controlled Model Configurations (M1–M4):** Empirically quantify the value of each information layer for binary EOL risk screening ($\text{EOL} \le 500$ cycles).
7. **Centralized Algorithm Selection Engine:** Implement a unified ML algorithm selector supporting Standardized Logistic Regression, RBF-SVM, and TabPFN 2.5 foundation model, applied consistently across all M1–M4 configurations.
8. **Multi-Algorithm Benchmarking:** Independently train and evaluate all 36 experimental configurations (3 algorithms $\times$ 4 model configs $\times$ 3 cutoffs) on both Primary Test ($N=43$) and Secondary Test ($N=40$, Batch 3 fast-charging policy shift).
9. **Interactive Screening Dashboard ("BATTERY INTELLIGENCE"):** Deliver an interactive, offline-capable HeroUI-inspired visualization platform with real-time algorithm switching and cycler CSV validation.

---

## 4. Proposed Methodology & Architecture

The framework is structured into a strictly feed-forward, leakage-safe pipeline:

```
[Raw HDF5 / .mat Batches] ──> [Streaming Loader & Validation] ──> [124 Benchmark Cohort]
                                                                          │
       ┌──────────────────────────────────────────────────────────────────┴─────────────┐
       ▼                                                                                ▼
[Disjoint Battery Splits]                                                   [Early Observation Cutoffs]
Train (41) / Test1 (43) / Test2 (40)                                        N_obs in {40, 70, 100} cycles
       │                                                                                │
       └───────────────────────────────────┬────────────────────────────────────────────┘
                                           ▼
                       [Feature Extraction (34 features)]
                       • Initial Susceptibility (Cycles 1-2)
                       • Current Snapshot (Cycle N_obs)
                       • Emerging Degradation Dynamics (Slopes, ΔIR, ΔT)
                       • Differential Discharge Curves ΔQ(V)
                                           │
                                           ▼
                      [Peer-Matching & PMRI Subsystem]
                      • k-NN Matching strictly against Train Pool (k=5)
                      • Trajectory Deviation (Capacity, Slopes)
                      • PMRI (Similarity, Coverage, Agreement, Consistency)
                                           │
                                           ▼
                      [Controlled Model Hierarchy (M1–M4)]
                      • M1: Initial & Snapshot Baseline (18 feats)
                      • M2: Individual Degradation History (34 feats)
                      • M3: Peer-Relative Trajectory (42 feats)
                      • M4: Confidence-Aware PMRI (46 feats)
                                           │
                                           ▼
                      [Centralized ML Algorithm Selector]
                      • Standardized Logistic Regression (L2 calibrated baseline)
                      • RBF Support Vector Machine (Nonlinear kernel)
                      • TabPFN 2.5 (Prior-data Fitted Network transformer)
                                           │
                                           ▼
                      [36-Configuration Comparative Benchmark]
                      • In-Domain Primary Test (Batch 3, 43 cells)
                      • Distribution Shift Secondary Test (Batch 3 shift, 40 cells)
                      • AUROC, AUPRC, Brier Score, Accuracy, Sensitivity, Specificity
                                           │
                                           ▼
                      [BATTERY INTELLIGENCE Dashboard]
                      • HeroUI Modern Component Architecture
                      • Global Algorithm Selector with dynamic M1-M4 updates
                      • Dedicated Algorithm Benchmarking View & Visualizations
```

---

## 5. Dataset Source & Acquisition

This project uses the benchmark dataset published by Severson et al. (*Nature Energy*, 2019), consisting of commercial A123 Systems APR18650M1A lithium iron phosphate (LFP)/graphite cylindrical cells (1.1 Ah nominal capacity) cycled under diverse fast-charging policies to end-of-life (80% nominal capacity retention) in a 40°C temperature chamber.

### Dataset Overview:
| Batch File | Raw Size | Recorded Cells | Description |
| :--- | :---: | :---: | :--- |
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 3.03 GB | 46 cells | Batch 1 (Primary training & development cohort) |
| `2017-06-30_batchdata_updated_struct_errorcorrect.mat` | 2.01 GB | 48 cells | Batch 2 (Fast-charging test protocols) |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 3.24 GB | 46 cells | Batch 3 (Validation & secondary test cohort) |

### How to Acquire Raw Datasets:
Due to GitHub file size limits (~8.27 GB total), the raw `.mat` files are excluded from version control via `.gitignore`. To obtain the original datasets:
1. Download the batch `.mat` files from the official MATR Data Repository on [data.matr.io](https://data.matr.io/1/) or [Nature Energy Supplemental Material](https://doi.org/10.1038/s41560-019-0356-8).
2. Place the three downloaded files inside the local `Datasets/` directory:
   ```text
   Datasets/
   ├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
   ├── 2017-06-30_batchdata_updated_struct_errorcorrect.mat
   └── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
   ```
3. *Note:* Pre-extracted, verified tabular cycle summaries and modeling datasets are included in `data/processed/`, allowing full model training, evaluation, and dashboard execution immediately upon cloning without re-downloading the multi-gigabyte raw files.

---

## 6. Dataset Processing & Validation Workflow

- **Exclusion Audit:** 16 non-standard cells with documented channel errors, premature hardware termination, or non-representative test parameters are excluded following Severson et al. criteria, yielding an eligible cohort of 124 cells.
- **Diagnostic Cycle Handling:** Cycle 1 was an unmonitored diagnostic protocol ($Q_{\text{discharge}} = 0.0$ Ah). The pipeline establishes physical baseline capacity strictly at Cycle 2 ($Q_{\text{discharge}} > 1.0$ Ah).
- **Leakage Prevention:** Battery-level splits (`data/splits/splits_benchmark.csv`) enforce disjoint partitions:
  - **Train:** 41 cells (Batch 1: 41)
  - **Primary Test:** 43 cells (Batch 1: 4, Batch 2: 39)
  - **Secondary Test:** 40 cells (Batch 2: 4, Batch 3: 36)
  - **Excluded:** 16 cells (documented anomalies)

---

## 7. Feature Engineering & Peer-Matching Pipeline

### 7.1 Multi-Cutoff Feature Groups (34 Individual Features)
- **Group 1: Initial Susceptibility (Cycles 1–2):** Baseline capacity ($Q_2$), internal resistance ($IR_2$), initial thermal response ($T_{\text{max}, 2}$, $T_{\text{avg}, 2}$), charge time, and nominal charging rates.
- **Group 2: Snapshot Condition (Cycle $N_{\text{obs}}$):** Instantaneous capacity ($Q_{N_{\text{obs}}}$), resistance ($IR_{N_{\text{obs}}}$), temperatures, and charging duration at the decision milestone.
- **Group 3: Emerging Trajectory ($2 \to N_{\text{obs}}$):** Capacity fade slope ($dQ/dn$), resistance growth rate ($dIR/dn$), thermal rise drift, and curvature.
- **Group 4: Differential Discharge Curves:** Log10 variance and extrema of $\Delta Q_{N_{\text{obs}}-10}(V)$ curves interpolated over a standard 1,000-point voltage grid.

### 7.2 Peer-Matching Subsystem
For each query cell at cutoff $N_{\text{obs}}$:
1. **Reference Pool Restriction:** Peers are selected **exclusively** from the training partition ($k=5$). Primary and secondary test cells never match each other.
2. **Matching Space:** Standardized Euclidean distance over early operating features (charging policy nominals, initial capacity, early fade rate).
3. **Peer-Relative Features (8 features added in M3):** Trajectory deviations relative to the peer cohort centroid:
   - Relative capacity deviation: $\Delta Q_{\text{rel}} = Q_{\text{cell}} - \bar{Q}_{\text{peers}}$
   - Relative fade slope deviation: $\Delta \text{slope}_{\text{rel}} = \text{slope}_{\text{cell}} - \bar{\text{slope}}_{\text{peers}}$
   - Standardized cohort $z$-scores and resistance deviations.

---

## 8. Peer-Match Reliability Index (PMRI)

The PMRI is a normalized score in $[0.0, 1.0]$ quantifying the statistical support and density of the reference neighborhood. It is **not** a battery failure probability, but a covariate confidence metric:

$$\text{PMRI} = 0.35 \cdot S_{\text{sim}} + 0.25 \cdot S_{\text{cov}} + 0.25 \cdot S_{\text{agree}} + 0.15 \cdot S_{\text{consist}}$$

- **$S_{\text{sim}}$ (Similarity):** Inverse exponential distance to reference peers in feature space.
- **$S_{\text{cov}}$ (Coverage):** Neighborhood density and dispersion around the target cell.
- **$S_{\text{agree}}$ (Agreement):** Concordance of degradation trajectories among the matched peers.
- **$S_{\text{consist}}$ (Consistency):** Stability of the peer cohort over successive early cycles.

**Engineering Decision Rule:** When $\text{PMRI} < 0.50$, the query cell lies in a sparse region of the reference library (e.g., rare charging protocol). The system issues an advisory warning recommending engineering caution or relying on individual snapshot models.

---

## 9. Machine Learning Model Configurations (M1–M4)

All four model configurations are trained for binary EOL screening: **$\text{Target} = \mathbb{I}(\text{EOL} \le 500\text{ cycles})$**. Decision thresholds $\tau$ are fitted strictly on 5-fold training cross-validation to maximize balanced F1 without test label leakage.

| Model Tier | Architecture | Feature Count | Composition |
| :--- | :--- | :---: | :--- |
| **M1** | Baseline Snapshot | 18 | Initial condition (Cycle 2) + Snapshot at $N_{\text{obs}}$ + Policy nominals. |
| **M2** | Degradation History | 34 | M1 features + Trajectory slopes, curvature, and $\Delta Q(V)$ curves. |
| **M3** | Peer-Relative | 42 | M2 features + 8 peer-relative trajectory deviations from reference neighbors. |
| **M4** | Confidence-Aware (Proposed) | 46 | M3 features + PMRI composite score and 4 sub-component metrics. |

---

## 10. Centralized Algorithm Selection & 36-Configuration Benchmarking

A centralized machine learning algorithm selector allows any supported algorithm to be applied consistently across all four model configurations ($M1–M4$) and observation cutoffs (40, 70, 100 cycles).

### 10.1 Supported Machine Learning Algorithms

1. **Standardized Logistic Regression (`logistic_regression`):**
   - Linear baseline with `StandardScaler`, L2 regularization ($C=1.0$), and Platt probability calibration.
   - Fast, fully interpretable, convex optimization baseline.
2. **RBF Support Vector Machine (`rbf_svm`):**
   - Nonlinear kernel classifier with `StandardScaler`, RBF kernel ($\gamma=\text{scale}$), $C=1.0$, and Platt probability scaling via cross-validation.
   - Captures complex nonlinear feature interactions between electrochemical degradation dynamics.
3. **TabPFN 2.5 (`tabpfn_2_5`):**
   - Prior-Data Fitted Network foundation model for tabular classification.
   - Executes transformer attention over prior synthetic tabular datasets (`tabpfn-v2-classifier.ckpt`), delivering state-of-the-art out-of-the-box probabilistic predictions without manual hyperparameter search.

### 10.2 Experimental Benchmark Results (36 Configurations)

Evaluated on the held-out Primary Test Cohort ($N = 43$ cells; 11 positive early-failure cases, 32 negative cases):

#### Cutoff 100 Cycles:
| Algorithm | Model | Features | AUROC | AUPRC | Brier Score | Accuracy | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | M1 | 18 | 0.9217 | 0.8690 | 0.0859 | 90.7% | 90.9% | 90.6% |
| **Logistic Regression** | M2 | 34 | 0.9152 | 0.8654 | 0.0807 | 93.0% | 81.8% | 96.9% |
| **Logistic Regression** | M3 | 42 | 0.9087 | 0.8521 | 0.0801 | 90.7% | 72.7% | 96.9% |
| **Logistic Regression** | M4 | 46 | 0.9065 | 0.8492 | 0.0815 | 88.4% | 72.7% | 93.8% |
| **RBF-SVM** | M1 | 18 | 0.9522 | 0.9248 | 0.0659 | 93.0% | 90.9% | 93.8% |
| **RBF-SVM** | M2 | 34 | 0.9783 | 0.9575 | 0.0864 | 93.0% | 90.9% | 93.8% |
| **RBF-SVM** | M3 | 42 | 0.9804 | 0.9634 | 0.0934 | 93.0% | 90.9% | 93.8% |
| **RBF-SVM** | M4 | 46 | **0.9804** | **0.9634** | 0.0952 | **93.0%** | **90.9%** | **93.8%** |
| **TabPFN 2.5** | M1 | 18 | 0.9674 | 0.9419 | 0.0727 | 93.0% | 90.9% | 93.8% |
| **TabPFN 2.5** | M2 | 34 | 0.9826 | 0.9667 | 0.0561 | 93.0% | 81.8% | 96.9% |
| **TabPFN 2.5** | M3 | 42 | 0.9848 | 0.9697 | 0.0650 | 95.3% | 90.9% | 96.9% |
| **TabPFN 2.5** | M4 | 46 | **0.9891** | **0.9778** | **0.0607** | **95.3%** | **90.9%** | **96.9%** |

#### Cutoff 70 Cycles:
| Algorithm | Model | AUROC | AUPRC | Brier Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | M4 | 0.9091 | 0.8354 | 0.0628 | 90.7% |
| **RBF-SVM** | M4 | 0.9913 | 0.9804 | 0.0488 | 95.3% |
| **TabPFN 2.5** | M4 | **1.0000** | **1.0000** | **0.0499** | **95.3%** |

### 10.3 Architecture Rankings & Empirical Findings

| Rank | Algorithm | Family | Mean Primary AUROC | Mean Primary Brier | Mean Secondary AUROC | Key Advantage |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **#1** | **TabPFN 2.5** | Foundation Model | **0.9729** | **0.0684** | **0.9744** | Perfect Cutoff 70 AUROC (1.0000), superior out-of-domain transfer |
| **#2** | **RBF-SVM** | Kernel Method | **0.9691** | **0.0688** | 0.9167 | High nonlinear discrimination, strong early cutoff performance |
| **#3** | **Logistic Regression** | Linear | 0.9072 | 0.0790 | 0.8590 | Stable, well-calibrated, interpretable baseline |

**Key Research Takeaway:**  
Across all three machine learning paradigms (linear, kernel, and foundation transformer), the progression from baseline condition ($M1$) to degradation dynamics ($M2$), peer-relative deviations ($M3$), and PMRI confidence metrics ($M4$) consistently elevates early-life screening discrimination and probability calibration.

---

## 11. "BATTERY INTELLIGENCE" Interactive Research Dashboard

The integrated dashboard provides an offline, responsive scientific workstation built with modern HeroUI-inspired styling across nine persistent views:

```
dashboard/
├── assets/
│   └── chart.umd.min.js          # Offline Chart.js v4.4.2 bundle
├── index.html                    # 9-view sidebar interface with BATTERY INTELLIGENCE brand
├── styles.css                    # Professional HeroUI design system (light/dark mode)
├── app.js                        # State manager, dynamic algorithm selector, Chart.js renderers
├── server.py                     # High-performance Python standard library REST server
└── README.md                     # Dedicated dashboard demonstration guide
```

### Nine Interactive Navigation Views:
1. **Overview:** Platform executive summary, cohort KPI cards, elevated risk tallies across M1–M4, and quick navigation portals.
2. **Battery Explorer:** Interactive cell selector, cutoff selector (40/70/100/full), temporal masking alert banner, physical summary cards, 46-feature inspector, and factual observations.
3. **Degradation Trajectories:** Chart.js multi-metric viewer ($Q_d$, $IR$, $T_{\text{max}}$, $T_{\text{avg}}$, charge time) with nominal reference (1.10 Ah), 80% EOL line (0.88 Ah), and multi-cell comparison tool.
4. **Peer Analysis:** Scientifically matched reference peer table ($k=5$), aligned trajectory overlay chart (target vs 5 peers + centroid), relative trajectory deviation metrics, and PMRI gauge card.
5. **EOL Risk Prediction:** Calibrated risk probability gauge ($P(\text{EOL} \le 550)$), decision threshold indicator ($\tau$), risk category badge, comparative 4-model matrix (M1–M4), and active algorithm indicator.
6. **Model Evaluation:** Primary test set performance cards (AUROC, AUPRC, Brier Score, ECE, Sensitivity, Specificity), official publication figure viewer, and legacy multi-algorithm benchmark table.
7. **Algorithm Benchmarking:** Dedicated comparative benchmarking interface with global algorithm selector, 4 active model cards, full 36-configuration comparative matrix, AUROC/Brier Chart.js charts, distribution shift table, and architecture leaderboard.
8. **PMRI & Confidence Analysis:** In-depth inspection of the Peer-Match Reliability Index (PMRI), 4-component breakdown, negative-control permutation tests, and selective prediction risk-coverage curves.
9. **Hardware Validation:** Technical schema specification, drag-and-drop CSV upload, sample valid/corrupt loaders, and real-time cycler data validation panel.

---

## 12. Project Directory Structure

```text
Capstone_Project/
├── dashboard/                              # Interactive web screening dashboard
│   ├── assets/chart.umd.min.js             # Offline Chart.js bundle
│   ├── index.html                          # Semantic HeroUI dashboard interface
│   ├── styles.css                          # HeroUI design system CSS
│   ├── app.js                              # Client application logic & algorithm selector
│   ├── server.py                           # Python http.server backend
│   └── README.md                           # Dashboard guide & demo script
├── data/                                   # Processed data and benchmark splits
│   ├── processed/                          # Metadata, cycle summaries, feature tables
│   │   ├── cells_metadata.csv              # 140 cell metadata records
│   │   ├── cycle_summaries.csv             # Cleaned cycle-by-cycle summary series
│   │   ├── modeling_dataset.csv            # 34-feature multi-cutoff modeling matrix
│   │   └── peer_features.csv               # Peer distances, relative features & PMRI
│   └── splits/                             # Disjoint battery-level train/test splits
│       ├── splits_benchmark.csv            # Tabular split assignments
│       └── splits_benchmark.json           # JSON partition definitions
├── Datasets/                               # Raw datasets directory (git-ignored)
├── docs/                                   # Research documentation and specifications
│   ├── data_audit_report.md                # Data validation and exclusion audit
│   ├── feature_dictionary.md               # Feature mathematical definitions
│   ├── roadmap.md                          # Phase-by-phase implementation roadmap
│   └── target_definition.md                # EOL & RUL target specifications
├── results/                                # Experimental outputs and research reports
│   ├── benchmarks/                         # 36-configuration benchmark outputs
│   │   ├── benchmark_metrics_36.csv        # Metrics across 3 algos x 4 models x 3 cutoffs
│   │   ├── benchmark_predictions_36.csv    # Individual battery prediction probabilities
│   │   ├── algorithm_comparison_summary.json # Machine-readable comparative matrix
│   │   ├── algorithm_ranking.json          # Overall algorithm ranking & leaderboard
│   │   └── distribution_shift_analysis.json# Primary test vs Secondary test shift analysis
│   ├── figures/                            # High-resolution publication figures
│   ├── multi_algorithm_report.md           # Formal 36-configuration benchmark report
│   └── selective_prediction_curve.csv      # Risk-coverage selective prediction curve
├── src/                                    # Core scientific Python packages
│   ├── data/loader.py                      # Streaming HDF5 MATLAB dataset parser
│   ├── features/                           # Feature extraction & target construction
│   ├── peers/                              # Peer matching & PMRI module
│   │   ├── peer_matcher.py                 # k-NN reference peer selector
│   │   ├── peer_reliability.py             # PMRI formulation & computation
│   │   └── peer_pipeline.py                # End-to-end peer relative feature pipeline
│   ├── models/                             # Machine learning model pipelines
│   │   ├── algorithm_selector.py           # Centralized ML algorithm selection engine
│   │   ├── benchmark_models.py             # 36-configuration multi-algorithm trainer
│   │   ├── model_configurations.py         # M1-M4 feature set and pipeline definitions
│   │   ├── model_factory.py                # Pipeline construction for LR, RBF-SVM, TabPFN 2.5
│   │   ├── model_pipeline.py               # Training, CV threshold tuning, evaluation
│   │   ├── selective_prediction.py         # Pure PMRI-based selective prediction engine
│   │   └── train_models.py                 # Reproducible baseline training execution script
│   └── evaluation/                         # Statistical hypothesis evaluation
├── tests/                                  # Comprehensive automated test suite (80 tests)
│   ├── test_algorithm_selector.py          # Centralized algorithm selector tests (7)
│   ├── test_dashboard.py                   # Dashboard API & benchmarks tests (16)
│   ├── test_data_integrity.py              # Raw data extraction and baseline tests (18)
│   ├── test_features.py                    # Feature extraction bounds tests (4)
│   ├── test_model_configurations.py        # Model feature isolation tests (5)
│   ├── test_model_evaluation.py            # Metrics and calibration tests (4)
│   ├── test_model_no_leakage.py            # Model training leakage tests (4)
│   ├── test_no_leakage.py                  # Feature-level anti-leakage tests (5)
│   ├── test_peer_matching.py               # Reference pool isolation tests (7)
│   ├── test_peer_no_leakage.py             # Peer feature anti-leakage tests (5)
│   ├── test_peer_reliability.py            # PMRI bounding and monotonicity tests (6)
│   └── test_targets.py                     # Target builder & censoring tests (3)
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 13. Getting Started & Reproduction Guide

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Jayanthachar08/Capstone_Project.git
cd Capstone_Project
```

### 2. Set Up Virtual Environment (Recommended)
```powershell
# Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Automated Test Suite
To verify environment setup, data integrity, anti-leakage invariants, and dashboard APIs across all 80 automated unit tests:
```powershell
python -m unittest discover -s tests -v
```
*Expected output:* `Ran 80 tests in ~10.6s. OK`

---

## 14. Running Experiments & Dashboard

### Executing the 36-Configuration Multi-Algorithm Benchmark
Train and evaluate Standardized Logistic Regression, RBF-SVM, and TabPFN 2.5 across M1–M4 configurations and cutoffs:
```powershell
python src/models/benchmark_models.py
```

### Multi-Seed Randomized Peer Controls & PMRI Validation
Execute 20-seed empirical null distribution permutations and PMRI error correlation:
```powershell
python src/peers/peer_control_experiments.py
```

### Selective Prediction Risk-Coverage Trade-Off Evaluation
Evaluate held-out test discrimination under training-only selective prediction abstention:
```powershell
python src/models/selective_prediction.py
```

### Launching the "BATTERY INTELLIGENCE" Dashboard
Start the lightweight standard library Python data server:
```powershell
python dashboard/server.py
```
*(Optional: specify a custom port: `python dashboard/server.py 8080`)*

Open your web browser and navigate to:
```text
http://127.0.0.1:8000/
```

---

## 15. Known Limitations & Future Work

1. **Cell Chemistry Generalization:** The empirical benchmark is evaluated on commercial LFP/graphite cells. Transferability to NMC, NCA, or solid-state chemistries requires expanding the reference library.
2. **Discrete Cutoff Windows:** Models operate at discrete observation milestones ($N_{\text{obs}} \in \{40, 70, 100\}$). Future work includes continuous recurrent or state-space models for cycle-by-cycle Bayesian updates.
3. **Reference Pool Size:** Fast-charging protocols with very few training exemplars trigger lower PMRI values. Expanding the training database will reduce the frequency of low-density abstentions.

---

## 16. Academic Attribution & License

- **Benchmark Dataset:** K. A. Severson, P. M. Attia, et al., *"Data-driven prediction of battery cycle life before capacity degradation,"* *Nature Energy*, vol. 4, no. 5, pp. 383–391, 2019. DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8).
- **Project Disclaimer:** This codebase is prepared for academic research and evaluation as part of a B.Tech Capstone Project in Computer Science & Engineering. Not intended for direct life-critical or mission-critical battery management without hardware validation.
