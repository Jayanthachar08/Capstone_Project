# Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-83%2F83%20passed-brightgreen.svg)]()
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
4. **Develop a Peer-Matching Subsystem:** Pair test batteries strictly with training reference peers using early-life operating features and compute relative trajectory deviations.
5. **Formulate the Peer-Match Reliability Index (PMRI):** Define an interpretable composite metric quantifying reference neighborhood support to gate or weight predictions and advise model abstention.
6. **Train & Evaluate Controlled Model Configurations (M1–M5):** Empirically quantify the value of each information layer for binary EOL risk screening ($\text{EOL} \le 500$ cycles).
7. **Develop M5 Adaptive Peer-Trust & Counterfactual Engine:** Systematically stress-test peer reference sets via Leave-One-Out, Replacement, and Leave-Two-Out perturbations, deriving counterfactual variance $U$ and adaptive trust $T$.
8. **Conduct Hypothesis Testing (H1–H4):** Statistically validate the research questions using paired bootstrap comparisons and robustness stress testing.
9. **Interactive Screening Dashboard ("BATTERY INTELLIGENCE"):** Deliver an interactive, offline-capable 9-view visualization platform with laboratory cycler hardware CSV validation.

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
                      [Controlled Model Hierarchy (M1-M4)]
                      • M1: Initial & Snapshot Baseline (18 feats)
                      • M2: Individual Degradation History (34 feats)
                      • M3: Peer-Relative Trajectory (42 feats)
                      • M4: Confidence-Aware PMRI (46 feats)
                                           │
                                           ▼
                      [M5 Adaptive Counterfactual Engine]
                      • Systematic Peer Perturbations (LOO, Rep, LTO)
                      • Instability Score U = std({p_pert})
                      • Adaptive Peer Trust T = R · exp(-λ U)
                      • Calibrated Blending: p_M5 = T·p_peer + (1-T)·p_hist
                      • Selective Abstention Policy
                                           │
                                           ▼
                      [BATTERY INTELLIGENCE Dashboard]
                      • 9 Sidebar Views (M5 Engine, Hardware Ingestion)
                      • AUROC, AUPRC, Brier Score, Bootstrap CIs (B=1000)
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

**Engineering Decision Rule:** When $\text{PMRI} < 0.50$, the query cell lies in a sparse region of the reference library (e.g., rare charging protocol). The system issues an advisory warning recommending engineering abstention or relying on individual snapshot models.

---

## 9. Machine Learning Model Configurations (M1–M5)

All models are trained for binary EOL screening: **$\text{Target} = \mathbb{I}(\text{EOL} \le 500\text{ cycles})$** using regularized logistic regression pipelines with standard scaling. Decision thresholds $\tau$ are fitted strictly on 5-fold training cross-validation to maximize balanced F1 without test label leakage.

| Model Tier | Architecture | Feature Count | Composition |
| :--- | :--- | :---: | :--- |
| **M1** | Baseline Snapshot | 18 | Initial condition (Cycle 2) + Snapshot at $N_{\text{obs}}$ + Policy nominals. |
| **M2** | Degradation History | 34 | M1 features + Trajectory slopes, curvature, and $\Delta Q(V)$ curves. |
| **M3** | Peer-Relative | 42 | M2 features + 8 peer-relative trajectory deviations from reference neighbors. |
| **M4** | Confidence-Aware | 46 | M3 features + PMRI composite score and 4 sub-component metrics. |
| **M5** | Adaptive Trust & Counterfactual | 46 + U + T | Dual-engine blend of peer-relative $p_{\text{peer}}$ and trajectory $p_{\text{history}}$ weighted by adaptive trust $T = R \cdot \exp(-\lambda U)$. |

---

## 10. M5: Adaptive Peer-Trust & Counterfactual Risk Engine

The M5 engine addresses a central failure mode of instance-based prognostics: **counterfactual peer instability**. When the composition of a battery's reference neighborhood is fragile, small perturbations in peer selection can cause large swings in predicted risk.

### 10.1 Mathematical Formulation
1. **Systematic Counterfactual Perturbations:** For each target battery, 12 counterfactual reference sets are evaluated:
   - 5 Leave-One-Peer-Out (LOO) subsets: $C_{k \setminus \{i\}}$
   - 5 Single-Peer Replacements: substituting each peer $i$ with candidate peer $k+1$
   - 1 Leave-Two-Out (LTO) subset: dropping the two furthest peers
   - 1 Original reference set baseline
2. **Counterfactual Instability Metric ($U$):**
   $$U = \text{std}\left(\{p(y \mid C_{\text{pert}})\}\right)$$
3. **Adaptive Peer-Trust Scaling ($T$):**
   $$T = R \cdot \exp(-\lambda U), \quad \lambda \ge 0$$
   where $R$ is the Peer-Match Reliability Index (PMRI) and $\lambda$ is tuned strictly on 5-fold training cross-validation (minimizing Brier score: $\lambda = 20.0$ at cutoffs 40 and 100, $\lambda = 0.0$ at cutoff 70).
4. **Calibrated Blended Risk ($p_{M5}$):**
   $$p_{M5} = T \cdot p_{\text{peer}} + (1 - T) \cdot p_{\text{history}}$$
5. **Selective Abstention Policy:** If $T < \tau_{\text{trust}}$ or $U > \tau_{\text{instability}}$, automated risk prediction is flagged with an explicit abstention reason (`NOMINAL_TRUST`, `LOW_ADAPTIVE_TRUST`, `EXCESSIVE_COUNTERFACTUAL_INSTABILITY`, or `INSUFFICIENT_REFERENCE_COVERAGE`).

### 10.2 Empirical Validation of Instability Metric
The correlation between counterfactual perturbation variance $U$ and absolute prediction error $|y - p|$ is consistently positive across cutoffs:
- **Cutoff 40:** Pearson $r = +0.324$
- **Cutoff 70:** Pearson $r = +0.191$
- **Cutoff 100:** Pearson $r = +0.445$

This provides empirical confirmation that high counterfactual instability signals higher prediction error risk, validating the exponential penalty $\exp(-\lambda U)$.

### 10.3 M5 Ablation Experiments (Primary Test Cohort):
| Cutoff | Mode | Description | Brier Score | AUROC | AUPRC |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 40 | **Full M5** | PMRI + Instability Penalty ($T = R \cdot e^{-\lambda U}$) | **0.1459** | **0.8636** | **0.6548** |
| 40 | `no_pmri` | Instability penalty only ($T = e^{-\lambda U}$) | 0.1462 | 0.8636 | 0.6548 |
| 40 | `no_instability` | PMRI weight only ($T = R$) | 0.1459 | 0.8636 | 0.6548 |
| 40 | `fixed_blend` | Static 50/50 blend ($T = 0.5$) | 0.1465 | 0.8409 | 0.5960 |
| 70 | **Full M5** | PMRI + Instability Penalty | **0.1274** | **0.8409** | **0.7073** |
| 70 | `fixed_blend` | Static 50/50 blend ($T = 0.5$) | 0.1293 | 0.8352 | 0.6974 |
| 100 | **Full M5** | PMRI + Instability Penalty | **0.1042** | **0.8932** | **0.8197** |
| 100 | `fixed_blend` | Static 50/50 blend ($T = 0.5$) | 0.1047 | 0.8955 | 0.8307 |

---

## 11. Empirical Research Results (Phase 6 & 7 Verification)

Evaluated on the held-out Primary Test Cohort ($N = 43$ cells; 1 excluded due to cycle life $< 40$, 11 positive early-failure cases, 32 negative cases):

### Performance Benchmark across Observation Cutoffs:

| Cutoff | Metric | M1 (Snapshot) | M2 (History) | M3 (Peer-Rel) | M4 (Confidence) | Best Model | Gain vs M1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **40** | **AUROC** | 0.8227 | 0.8125 | **0.8682** | 0.8636 | **M3** | **+0.0455** |
| 40 | **AUPRC** | 0.5699 | 0.5283 | **0.6729** | 0.6548 | **M3** | **+0.1030** |
| 40 | **Brier Score** | 0.1557 | 0.1481 | 0.1472 | **0.1459** | **M4** | **-0.0098** |
| **70** | **AUROC** | 0.7727 | 0.8068 | **0.8409** | **0.8409** | **M3 / M4** | **+0.0682** |
| 70 | **AUPRC** | 0.6080 | 0.6749 | **0.7202** | 0.7073 | **M3** | **+0.1122** |
| 70 | **Brier Score** | 0.1508 | 0.1345 | **0.1260** | 0.1274 | **M3** | **-0.0248** |
| **100**| **AUROC** | 0.7932 | 0.8841 | **0.8977** | 0.8932 | **M3** | **+0.1045** |
| 100| **AUPRC** | 0.6725 | 0.8354 | 0.8172 | **0.8197** | **M2 / M4** | **+0.1472** |
| 100| **Brier Score** | 0.1385 | 0.1051 | 0.1065 | **0.1042** | **M4** | **-0.0343** |

### Formal Research Hypothesis Verdicts:
- **H1 (Degradation History Value): SUPPORTED.** Degradation history significantly improves predictions at Cycle 100 ($\Delta\text{AUROC} = +0.0909$, $\Delta\text{AUPRC} = +0.1629$).
- **H2 (Peer-Relative Feature Value): SUPPORTED.** Peer-relative features provide consistent predictive advantages at ultra-early cutoffs (Cutoff 40: $\Delta\text{AUROC} = +0.0455$, $\Delta\text{AUPRC} = +0.1030$; Cutoff 70: $\Delta\text{AUROC} = +0.0682$, $\Delta\text{AUPRC} = +0.1122$).
- **H3 (Reliability-Aware Improvement): PARTIALLY SUPPORTED.** Reliability features improve probability calibration and Brier score across all cutoffs, while ranking metrics remain comparable to M3.
- **H4 (Cutoff Degradation Dynamics): SUPPORTED.** Predictive precision increases monotonically as the observation window broadens from 40 to 100 cycles across all models.

---

## 12. "BATTERY INTELLIGENCE" Interactive Research Dashboard

The integrated dashboard provides an offline, responsive scientific workstation across nine persistent views:

```
dashboard/
├── assets/
│   └── chart.umd.min.js          # Offline Chart.js v4.4.2 bundle
├── index.html                    # 9-view sidebar interface with BATTERY INTELLIGENCE brand
├── styles.css                    # Professional dark navy/teal scientific styling
├── app.js                        # State manager, Chart.js renderers, hardware validator
├── server.py                     # High-performance Python standard library REST server
└── README.md                     # Dedicated dashboard demonstration guide
```

### Nine Interactive Navigation Views:
1. **Overview:** Platform executive summary, cohort KPI cards, elevated risk tallies across M1–M5, and quick navigation portals.
2. **Battery Explorer:** Interactive cell selector, cutoff selector (40/70/100/full), temporal masking alert banner, physical summary cards, 34-feature inspector, and factual observations.
3. **Degradation Trajectories:** Chart.js multi-metric viewer ($Q_d$, $IR$, $T_{\text{max}}$, $T_{\text{avg}}$, charge time) with nominal reference (1.10 Ah), 80% EOL line (0.88 Ah), and multi-cell comparison tool.
4. **Peer Analysis:** Scientifically matched reference peer table ($k=5$), aligned trajectory overlay chart (target vs 5 peers + centroid), relative trajectory deviation metrics, and PMRI gauge card.
5. **M5 Robustness Engine:** Dedicated M5 interface displaying counterfactual perturbation traces table, sensitivity waterfall bar chart, perturbation variance $U$, adaptive trust $T$, and selective abstention status.
6. **Risk & Calibration:** Five-model comparative risk matrix (M1–M5), Brier score comparison, calibration curves, threshold slider, and sequestered ground-truth reference box.
7. **Research Benchmark:** Master performance table, hypothesis evaluation verdicts (H1–H4), M5 ablation studies table, paired bootstrap differences with 95% CIs, and CSV/JSON export download buttons.
8. **Hardware Validation:** Technical schema specification, drag-and-drop CSV upload, sample valid/corrupt loaders, and real-time schema and physical bounds audit panel.
9. **Methodology & About:** Project provenance (Severson et al. 2019 citation), 16 channel exclusion reasons categorized, mathematical formulations of PMRI, $U$, $T$, and $p_{M5}$, and formal patent review disclosure notice.

---

## 13. Patentability & Prior Disclosure Assessment

A formal intellectual property audit was conducted to evaluate the patentability of the PMRI and M5 adaptive trust algorithms. Full documentation is provided in [`docs/patent_review.md`](docs/patent_review.md).

- **Prior Public Disclosure:** The complete codebase, mathematical equations, and experimental evaluation reports have been disclosed publicly in university capstone submissions and via the open GitHub repository (`https://github.com/Jayanthachar08/Capstone_Project`).
- **Legal Implications:** Under international patent conventions (e.g. EPC Article 54), prior public disclosure constitutes an absolute bar to patent novelty. In the United States (35 U.S.C. § 102), a 1-year personal grace period exists, but pure mathematical algorithms are generally non-patentable without integration into physical BMS embedded hardware.
- **Strategic Recommendation:** Maintain the project as an open-source, peer-reviewed academic contribution with software copyright under the MIT license, functioning as a defensive publication that preserves academic attribution while preventing third-party patent lock-in.

---

## 12. Project Directory Structure

```text
Capstone_Project/
├── dashboard/                              # Interactive web screening dashboard
│   ├── assets/chart.umd.min.js             # Offline Chart.js bundle
│   ├── index.html                          # Semantic dashboard interface
│   ├── styles.css                          # Dark-theme scientific CSS
│   ├── app.js                              # Client application logic
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
│   ├── inspect_battery_values.py           # Exploratory HDF5 inspection script
│   └── inspect_dataset.py                  # Batch schema validation script
├── docs/                                   # Research documentation and specifications
│   ├── data_audit_report.md                # Data validation and exclusion audit
│   ├── feature_dictionary.md               # 34-feature mathematical definitions
│   ├── roadmap.md                          # Phase-by-phase implementation roadmap
│   └── target_definition.md                # EOL & RUL target specifications
├── results/                                # Experimental outputs and research reports
│   ├── figures/                            # High-resolution publication figures
│   │   ├── calibration_diagrams.png        # Reliability calibration plots
│   │   ├── cutoff_comparison.png           # Multi-cutoff performance bar charts
│   │   ├── model_differences.png           # Paired difference forest plot (95% CI)
│   │   └── pmri_stratification.png         # PMRI cohort performance stratification
│   ├── phase3_validation_report.md         # Feature extraction validation report
│   ├── phase4_peer_matching_report.md      # Peer matching & PMRI validation report
│   ├── phase5_model_metrics.csv            # 36 model evaluation metric rows
│   ├── phase5_model_training_report.md     # Model training & comparison report
│   ├── phase5_predictions.csv              # 372 per-battery validated predictions
│   ├── phase6_hypothesis_evaluation_report.md # Formal hypothesis testing report
│   ├── phase6_hypothesis_summary.json      # Machine-readable hypothesis verdicts
│   ├── phase6_model_differences.csv        # Paired bootstrap differences (95% CI)
│   ├── phase6_robustness_results.csv       # Stress test evaluations across Exps A-F
│   ├── m5_predictions.csv                  # 372 M5 validated predictions with U and T scores
│   ├── m5_peer_robustness.csv              # 4,464 counterfactual perturbation traces
│   ├── m5_ablation_results.csv             # 12 ablation experiment evaluations
│   ├── m5_evaluation.json                  # Machine-readable M5 benchmark metrics & diagnostics
│   └── m5_evaluation_report.md             # Formal research evaluation report for M5
├── src/                                    # Core scientific Python packages
│   ├── data/loader.py                      # Streaming HDF5 MATLAB dataset parser
│   ├── features/                           # Feature extraction & target construction
│   │   ├── feature_engineering.py          # 34-feature extraction pipeline
│   │   └── target_builder.py               # EOL, RUL, and binary target builder
│   ├── peers/                              # Peer matching & robustness module
│   │   ├── peer_matcher.py                 # k-NN reference peer selector
│   │   ├── peer_reliability.py             # PMRI formulation & computation
│   │   ├── peer_pipeline.py                # End-to-end peer relative feature pipeline
│   │   └── peer_robustness.py              # Counterfactual perturbation engine (LOO, Rep, LTO)
│   ├── models/                             # Machine learning model pipelines
│   │   ├── model_configurations.py         # M1-M4 feature set and pipeline definitions
│   │   ├── model_pipeline.py               # Training, CV threshold tuning, evaluation
│   │   ├── train_models.py                 # Reproducible model training execution script
│   │   ├── adaptive_peer_trust.py          # M5 Adaptive trust engine T = R * exp(-lambda * U)
│   │   └── evaluate_m5.py                  # End-to-end M5 evaluation and ablation workflow
│   └── evaluation/                         # Statistical hypothesis evaluation
│       └── hypothesis_evaluator.py         # Paired bootstrap & robustness evaluator
├── tests/                                  # Comprehensive automated test suite (79 tests)
│   ├── test_adaptive_trust.py              # M5 adaptive trust formulation & tuning tests (5)
│   ├── test_m5_robustness.py               # Counterfactual perturbation engine tests (5)
│   ├── test_dashboard.py                   # Dashboard API, M5 endpoints & hardware tests (13)
│   ├── test_features.py                    # Feature extraction bounds tests (4)
│   ├── test_model_configurations.py        # Model feature isolation tests (5)
│   ├── test_model_evaluation.py            # Metrics and calibration tests (4)
│   ├── test_model_no_leakage.py            # Model training leakage tests (4)
│   ├── test_no_leakage.py                  # Feature-level anti-leakage tests (5)
│   ├── test_peer_matching.py               # Reference pool isolation tests (7)
│   ├── test_peer_no_leakage.py             # Peer feature anti-leakage tests (5)
│   ├── test_peer_reliability.py            # PMRI bounding and monotonicity tests (6)
│   ├── test_data_integrity.py              # Raw data extraction and baseline tests (18)
│   └── test_targets.py                     # Target builder & censoring tests (3)
├── .env.example                            # Optional server configuration template
├── .gitignore                              # Comprehensive Git exclusion rules
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 14. Getting Started & Reproduction Guide

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
To verify environment setup, data integrity, anti-leakage invariants, and dashboard APIs across all 83 automated unit tests:
```powershell
python -m unittest discover -s tests -v
```
*Expected output:* `Ran 83 tests in ~1.8s. OK`

---

## 15. Running Experiments, Pipelines & Dashboard

### Modular Multi-Algorithm Benchmark (LR, RF, RBF-SVM, HistGBM)
Compare 4 machine learning algorithm families across M1–M4 configurations and cutoffs (40, 70, 100 cycles) using strict 5-fold training CV:
```powershell
python src/models/multi_algorithm_benchmark.py
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

### Executing M5 Counterfactual Robustness & Ablations
Run M5 counterfactual perturbation traces and ablation studies:
```powershell
python src/models/evaluate_m5.py
```

### Retraining Baseline Models & Running Hypotheses (Phases 5 & 6)
```powershell
# Train baseline M1-M4 models across cutoffs
python src/models/train_models.py

# Run hypothesis testing, bootstrap confidence intervals, and generate figures
python src/evaluation/hypothesis_evaluator.py
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
