# Battery Degradation Screening & EOL Risk Prediction Dashboard

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Degree:** B.Tech Computer Science & Engineering (Capstone Project)  
**Environment:** Windows • Python 3.11 • Offline Capable HTML5/CSS/Vanilla JS  
**Current Phase:** Phase 7 — Machine Learning Models & Research Results Integration  

---

## 1. Overview & Purpose

This interactive scientific dashboard is designed for comprehensive demonstration to academic project guides, faculty evaluators, and battery researchers. It brings together all research outcomes from Phases 1 through 7 into a single, unified interface:

1. **Observed Battery Degradation Trajectories:** Early-life cycling measurements ($Q_{\text{discharge}}$, internal resistance, temperature, charge time) from the MATR/Severson et al. benchmark dataset.
2. **Early Observation Cutoffs:** Interactive exploration across early-life observation windows ($N_{\text{obs}} \in \{40, 70, 100\}$ cycles or Full Lifecycle) with strict temporal leakage prevention.
3. **Validated Machine Learning EOL Risk Models (M1–M4):** Live inspection of predictions across four model tiers:
   - **M1 (Baseline):** 18 Initial & snapshot features.
   - **M2 (Degradation History):** 34 Features (M1 + trajectory rates and $\Delta Q(V)$ differential capacity curves).
   - **M3 (Peer-Relative):** 42 Features (M2 + peer-relative deviations from reference neighbors).
   - **M4 (Full Proposed Framework):** 46 Features (M3 + Peer-Match Reliability Index components).
4. **Peer Trajectory Centroid Overlay:** Side-by-side visualization of a target battery against its top-5 reference peers and the computed peer centroid trajectory.
5. **Peer-Match Reliability Index (PMRI):** Interpretable reliability assessment broken down into its four constituent sub-indices: Similarity ($S_{\text{sim}}$), Coverage ($S_{\text{cov}}$), Agreement ($S_{\text{agree}}$), and Internal Consistency ($S_{\text{consist}}$), alongside clear advisory cautions when peer density is low.
6. **Empirical Research Findings (Phase 6):** Embedded research evaluation tabulating benchmark metrics across cutoffs, formal hypothesis testing verdicts (H1–H4), paired bootstrap difference confidence intervals, and high-resolution publication-grade figures.

---

## 2. Architecture & Tech Stack

```text
dashboard/
├── assets/
│   └── chart.umd.min.js          # Bundled offline Chart.js v4.4.2 (Zero internet dependency)
├── index.html                    # Semantic HTML5 scientific dashboard interface with Phase 7 UI
├── styles.css                    # Professional dark navy/blue theme, teal accents, responsive layout
├── app.js                        # Client-side state manager, API client, and Chart.js renderer
├── server.py                     # Lightweight Python http.server REST API serving real CSV & JSON artifacts
└── README.md                     # Operational documentation and demonstration guide
```

- **Zero Heavy Framework Dependencies:** Built entirely using Python's standard library (`http.server`, `csv`, `json`, `pathlib`) and vanilla ES6+ JavaScript. No Node.js, npm, React, Flask, or Django needed.
- **100% Offline Capable:** Runs without active internet connection; all assets and figures are served locally.
- **Fast In-Memory Queries:** Pre-indexes the processed datasets and model prediction artifacts on startup in $\sim 0.2$ seconds, responding to trajectory and feature queries in $<10$ milliseconds.
- **Strict Data Segregation:** Observed cycle data up to cutoff is strictly demarcated from retrospective ground truth labels.

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

The dashboard backend exposes standard REST endpoints returning JSON and static image artifacts:

| Endpoint | Method | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/api/overview` | `GET` | None | Returns high-level dataset statistics, model count, total predictions, and elevated risk tallies per cutoff/model. |
| `/api/batteries` | `GET` | None | Lists all 140 cells with metadata (batch, policy, lifecycle, split partition, exclusion status). |
| `/api/battery/<cell_id>` | `GET` | `cutoff` (40, 70, 100, all) | Returns cycle history, summary metrics, engineered features, sequestered ground truth, and M1–M4 predictions for the specified cell. |
| `/api/models/predict` | `GET` | `cell_id`, `cutoff` (40, 70, 100), `model` (M1, M2, M3, M4) | Returns detailed model prediction score, threshold, binary risk classification, PMRI components, and top peers for a cell. |
| `/api/peers/trajectories`| `GET` | `cell_id`, `cutoff` (40, 70, 100), `metric` | Returns aligned trajectories for the target cell, top-5 matched peers, the mean peer centroid curve, and trajectory deviations. |
| `/api/compare` | `GET` | `ids` (comma-separated), `metric`, `cutoff` | Multi-cell comparative trajectories aligned by cycle number. |
| `/api/research/summary` | `GET` | None | Returns Phase 6 research benchmark metrics table and formal hypothesis verdicts (H1–H4). |
| `/api/research/differences`| `GET`| None | Returns pairwise model metric differences with 95% bootstrap confidence intervals across cutoffs. |
| `/api/research/robustness` | `GET`| None | Returns robustness experiment evaluations across data variations (Experiments A–F). |
| `/results/figures/<file>`| `GET` | filename | Serves high-resolution research figures (e.g., `cutoff_comparison.png`, `model_differences.png`). |

---

## 5. Machine Learning Models (M1–M4) Summary

| Model Tier | Name | Features | Core Predictive Information |
| :--- | :--- | :---: | :--- |
| **M1** | Baseline Snapshot | 18 | Initial cycle health (Cycle 2 baseline), early snapshots, charging protocol nominals. |
| **M2** | Degradation History | 34 | Adds degradation rates (slopes, curvature, variance) and early $\Delta Q_{10-2}(V)$ differential capacity curves. |
| **M3** | Peer-Relative | 42 | Adds peer-relative deviation features ($z$-scores and differences from reference cohort neighbors). |
| **M4** | Confidence-Aware | 46 | Adds Peer-Match Reliability Index (PMRI) and component scores ($S_{\text{sim}}$, $S_{\text{cov}}$, $S_{\text{agree}}$, $S_{\text{consist}}$). |

**Primary Prediction Target:** `target_fail_within_500` (Binary classification: whether cell EOL $\le 500$ cycles).  
**Decision Thresholds:** Fitted on 5-fold training cross-validation (Cutoff 40: 0.315–0.345, Cutoff 70: 0.301–0.355, Cutoff 100: 0.310–0.360) to optimize balanced F1-score without test data leakage.

---

## 6. Peer-Match Reliability Index (PMRI) Interpretation

The PMRI is a normalized score in $[0.0, 1.0]$ measuring the **statistical density and quality of the reference peer neighborhood** in the training cohort, not the battery failure probability.

- **Similarity ($S_{\text{sim}}$):** Proximity in charging protocol and early degradation space.
- **Coverage ($S_{\text{cov}}$):** Distribution and spread of reference candidates around the target.
- **Agreement ($S_{\text{agree}}$):** Concordance of degradation rates among top peers.
- **Consistency ($S_{\text{consist}}$):** Variance of trajectory trajectories within the peer neighborhood.

### Advisory Interpretation:
- **High PMRI ($\ge 0.70$):** High confidence in peer-relative context. Reference neighbors closely reflect operating characteristics.
- **Moderate PMRI ($0.50 - 0.69$):** Acceptable peer support. Predictions should be interpreted with standard engineering tolerance.
- **Low PMRI ($< 0.50$):** **Advisory caution triggered.** The battery is operating in a sparse region of the training feature space (e.g., rare charging protocol). The dashboard advises engineering abstention or relying primarily on M2 snapshot models.

---

## 7. Faculty Demonstration Walkthrough

When demonstrating the project to your guide or evaluation committee, follow this logical flow:

1. **System Overview (Header & Overview Cards):**
   - Point out the active benchmark cohort (124 modeled cells, 16 excluded with documented anomalies).
   - Highlight the 4 ML models evaluated across 3 observation cutoffs (372 validated predictions total).
2. **Selecting a Battery & Cutoff:**
   - Select a primary test cell, e.g., `b2c1` (Batch 2, fast charging).
   - Toggle cutoff between **Cycle 40**, **Cycle 70**, and **Cycle 100**.
   - Note how the observation window adjusts dynamically while future cycle history is sequestered.
3. **ML Risk Prediction Evaluation (Section H):**
   - Examine the 4-model comparison table. Show how M1, M2, M3, and M4 predict risk probabilities.
   - Point out the decision thresholds and resulting risk classification (`ELEVATED RISK` vs `NOMINAL RISK`).
   - Compare with the sequestered ground truth box at the bottom of the card.
4. **Peer Trajectory Centroid Overlay (Section I):**
   - Show the target battery's degradation curve overlaid with its top-5 matched peers.
   - Point out the dashed **Mean Peer Centroid** trajectory and the deviation metric.
5. **Peer-Match Reliability Gauge (Section J):**
   - Explain the 4 component progress bars ($S_{\text{sim}}$, $S_{\text{cov}}$, $S_{\text{agree}}$, $S_{\text{consist}}$).
   - Highlight the scientific caution banner explaining that PMRI measures match quality and flags when the model should abstain.
6. **Research Model-Comparison Results (Section K):**
   - Switch between the four research tabs:
     - **Performance Benchmark:** Display AUROC, AUPRC, and Brier score across cutoffs.
     - **Hypothesis Verdicts:** Review formal findings for H1, H2, H3, and H4.
     - **Paired Bootstrap Differences:** Examine the 95% confidence intervals showing where peer-relative modeling provides statistically meaningful gains.
     - **Publication Figures:** Browse high-resolution plots generated during Phase 6 experiments.

---

## 8. Limitations & Scope

1. **Retrospective Cohort:** Evaluated on the 124-cell benchmark dataset from Severson et al. (commercial LFP/graphite cells). Generalization to NMC/NCA chemistries requires expanding the reference database.
2. **Observation Cutoffs:** Predictions are formulated at discrete early-life milestones ($N_{\text{obs}} \in \{40, 70, 100\}$ cycles). Continuous per-cycle inference can be evaluated in prospective online systems.
3. **Reference Pool Size:** Small cohort size in specific charging regimes can yield low PMRI scores, which the framework transparently flags rather than masking with overconfident predictions.
