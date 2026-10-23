# BATTERY INTELLIGENCE — Scientific Research Dashboard

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Degree:** B.Tech Computer Science & Engineering (Capstone Project)  
**Environment:** Windows • Python 3.11 • Offline Capable HTML5/CSS/Vanilla JS  
**Current Phase:** Phase 7+ — M5 Adaptive Counterfactual Engine & Professional Dashboard Integration  

---

## 1. Overview & Purpose

**BATTERY INTELLIGENCE** is an interactive research workstation designed for academic faculty, thesis evaluators, and battery prognostics engineers. It unifies all research outcomes from data extraction, leak-free feature engineering, reference peer matching, reliability indexing, five machine learning model tiers (M1–M5), and experimental validation:

1. **Observed Battery Degradation Trajectories:** Early-life cycling measurements ($Q_{\text{discharge}}$, internal resistance, surface temperature, charge duration) from the benchmark dataset by Severson et al. (*Nature Energy* 2019).
2. **Early Observation Cutoffs:** Interactive exploration across early observation windows ($N_{\text{obs}} \in \{40, 70, 100\}$ cycles or Full Lifecycle) with strict temporal isolation.
3. **Five Machine Learning EOL Risk Models (M1–M5):** Live inspection of predictions across five model configurations:
   - **M1 (Condition Baseline):** 18 Initial & snapshot features.
   - **M2 (Degradation History):** 34 Features (M1 + trajectory rates and differential capacity $\Delta Q(V)$).
   - **M3 (Peer-Relative):** 42 Features (M2 + peer-relative deviations from reference neighbors).
   - **M4 (PMRI-Augmented):** 46 Features (M3 + Peer-Match Reliability Index components).
   - **M5 (Adaptive Trust & Counterfactual Engine):** Adaptive blending of peer and trajectory predictions weighted by counterfactual perturbation variance $U$ and trust $T = R \cdot \exp(-\lambda U)$.
4. **Peer Trajectory Centroid Overlay:** Side-by-side visualization of a target battery against its top-5 reference peers and the computed peer centroid trajectory curve $\mu_{\text{peer}}(t)$.
5. **Peer-Match Reliability Index (PMRI):** Interpretable reliability assessment broken down into its four constituent sub-indices: Similarity ($S_{\text{sim}}$), Coverage ($S_{\text{cov}}$), Agreement ($S_{\text{agree}}$), and Consistency ($S_{\text{consist}}$).
6. **M5 Robustness Engine Interface:** Dedicated tab displaying systematic counterfactual perturbations (Leave-One-Out, Single-Peer Replacement, Leave-Two-Out), sensitivity waterfall charts, and selective abstention status.
7. **Research Benchmark:** Master performance table, hypothesis evaluation verdicts (H1–H4), paired bootstrap confidence intervals (B=1,000), ablation studies, and downloadable CSV/JSON exports.
8. **Laboratory Hardware Ingestion Protocol:** Technical schema specification and interactive validation interface prepared for external laboratory cycler data ingestion (verifying monotonicity, capacity bounds, and temperature limits).

---

## 2. Architecture & File Structure

```text
dashboard/
├── assets/
│   └── chart.umd.min.js          # Bundled offline Chart.js v4.4.2 (Zero external CDN dependency)
├── index.html                    # 9-view sidebar interface with BATTERY INTELLIGENCE brand
├── styles.css                    # Professional dark navy/teal scientific workstation stylesheet
├── app.js                        # Client application, 9-tab state manager, Chart.js renderers
├── server.py                     # Lightweight Python http.server REST API serving real CSV & JSON artifacts
└── README.md                     # Operational documentation and API reference
```

- **Zero Heavy Framework Dependencies:** Built entirely using Python's standard library (`http.server`, `csv`, `json`, `pathlib`) and vanilla ES6+ JavaScript. No Node.js, npm, React, Flask, or Django needed.
- **100% Offline Capable:** Runs without active internet connection; all assets and figures are served locally.
- **Fast In-Memory Queries:** Pre-indexes the processed datasets and model prediction artifacts on startup in $\sim 0.2$ seconds, responding to trajectory and feature queries in $<10$ milliseconds.
- **Strict Anti-Leakage Isolation:** Observed cycle data up to cutoff is strictly demarcated from retrospective ground truth labels.

---

## 3. How to Run the Dashboard

Open PowerShell or Command Prompt in the project workspace:

```powershell
cd "C:\engg notes\7th sem\Capstone_Project"
python dashboard/server.py
```

To run on an alternate port (e.g. 8080):
```powershell
python dashboard/server.py 8080
```

### Accessing the Dashboard
Open your web browser (Chrome, Edge, Firefox, or Safari) and navigate to:
```text
http://127.0.0.1:8000/
```

To stop the server gracefully, press `Ctrl + C` in the PowerShell terminal.

---

## 4. REST API Reference Endpoints

The dashboard backend exposes standard REST endpoints returning JSON, CSV, and static image artifacts:

| Endpoint | Method | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/api/overview` | `GET` | None | Returns high-level dataset statistics, model count (5), total predictions (372), and elevated risk tallies per cutoff/model. |
| `/api/batteries` | `GET` | None | Lists all 140 cells with metadata (batch, policy, lifecycle, split partition, exclusion status). |
| `/api/battery` | `GET` | `id`, `cutoff` (40, 70, 100, all) | Returns cycle history, summary metrics, engineered features, sequestered ground truth, and M1–M5 predictions for the specified cell. |
| `/api/models/predict` | `GET` | `id`, `cutoff`, `model` (M1–M5) | Returns detailed model prediction score, threshold, binary risk classification, PMRI components, and top peers for a cell. |
| `/api/peers/trajectories`| `GET` | `id`, `cutoff` (40, 70, 100) | Returns aligned trajectories for target cell, top-5 matched peers, mean peer centroid curve, and trajectory deviations. |
| `/api/m5/robustness` | `GET` | `id`, `cutoff` (40, 70, 100) | Returns M5 counterfactual perturbation traces, instability score $U$, adaptive trust $T$, sensitivity ranking, and abstention status. |
| `/api/m5/ablations` | `GET` | None | Returns M5 ablation evaluation metrics across cutoffs (comparing full, no_pmri, no_instability, fixed_blend). |
| `/api/m5/summary` | `GET` | None | Returns JSON summary of M5 parameter tuning ($\lambda$), threshold optimization, and error correlations. |
| `/api/export/benchmark` | `GET` | `format` (`csv` or `json`) | Downloads complete benchmark predictions and evaluation results as an attachment. |
| `/api/hardware/validate`| `POST`| Body: CSV string | Validates external laboratory cycler telemetry CSV against schema and physical bounds rules. |
| `/api/research/summary` | `GET` | None | Returns Phase 6 research benchmark metrics table and formal hypothesis verdicts (H1–H4). |
| `/api/research/differences`| `GET`| None | Returns pairwise model metric differences with 95% bootstrap confidence intervals across cutoffs. |
| `/results/figures/<file>`| `GET` | filename | Serves high-resolution research figures (e.g., `cutoff_comparison.png`, `model_differences.png`). |

---

## 5. Nine Interactive Navigation Views

1. **Overview:** High-level platform summary, cohort KPI metrics (140 total, 124 eligible, 16 excluded), elevated risk breakdown across M1–M5, and quick navigation portals.
2. **Battery Explorer:** Interactive cell selector, cutoff selector (40/70/100/full), temporal masking alert banner, physical summary cards, 34-feature inspector table, and factual trajectory observations.
3. **Degradation Trajectories:** Chart.js multi-metric viewer (Discharge Capacity, IR, Max Temp, Avg Temp, Charge Time), nominal reference line (1.10 Ah), 80% EOL line (0.88 Ah), cutoff marker, and multi-cell comparison tool.
4. **Peer Analysis:** Scientifically matched reference peer table ($k=5$), aligned trajectory overlay chart (target vs 5 peers + centroid), relative trajectory deviation metrics, and PMRI gauge card.
5. **M5 Robustness Engine:** Dedicated M5 interface displaying counterfactual perturbation traces table, sensitivity waterfall bar chart, perturbation variance $U$, adaptive trust $T$, and selective abstention status.
6. **Risk & Calibration:** Five-model comparative risk matrix (M1–M5), Brier score comparison, calibration curves, threshold slider, and sequestered ground-truth reference box.
7. **Research Benchmark:** Master performance table, hypothesis evaluation verdicts (H1–H4), M5 ablation studies table, paired bootstrap differences with 95% CIs, and CSV/JSON export download buttons.
8. **Hardware Validation:** Technical schema specification, drag-and-drop CSV upload, sample valid/corrupt loaders, and real-time schema and physical bounds audit panel.
9. **Methodology & About:** Project provenance (Severson et al. 2019 citation), 16 channel exclusion reasons categorized, mathematical formulations of PMRI, $U$, $T$, and $p_{M5}$, and formal patent review disclosure notice.
