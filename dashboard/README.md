# BATTERY HEALTH INTELLIGENCE — HeroUI Research Dashboard

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Degree:** B.Tech Computer Science & Engineering (Capstone Project)  
**Environment:** Windows • Python 3.11 • Offline-Capable HTML5 / HeroUI Design Tokens / Vanilla ES6+ JS  
**UI/UX Standard:** Professional SaaS Analytics & Scientific Workstation inspired by [HeroUI](https://heroui.com)  

---

## 1. Overview & Purpose

**BATTERY HEALTH INTELLIGENCE** is an interactive research workstation designed for academic faculty, thesis evaluators, and battery prognostics engineers. It unifies all research outcomes from data extraction, leak-free feature engineering, reference peer matching, reliability indexing, four machine learning model configurations (M1–M4), centralized multi-algorithm selection, and experimental validation:

1. **Observed Battery Degradation Trajectories:** Early-life cycling measurements ($Q_{\text{discharge}}$, internal resistance, surface temperature, charge duration) from the benchmark dataset by Severson et al. (*Nature Energy* 2019).
2. **Early Observation Cutoffs:** Interactive exploration across early observation windows ($N_{\text{obs}} \in \{40, 70, 100\}$ cycles or Full Lifecycle) with strict temporal isolation.
3. **Four Machine Learning EOL Risk Models (M1–M4):** Live inspection of predictions across four model configurations:
   - **M1 (Condition Baseline):** 18 Initial & snapshot features.
   - **M2 (Degradation History):** 34 Features (M1 + trajectory rates and differential capacity $\Delta Q(V)$).
   - **M3 (Peer-Relative):** 42 Features (M2 + peer-relative deviations from reference neighbors).
   - **M4 (PMRI-Aware):** 46 Features (M3 + Peer-Match Reliability Index components).
4. **Centralized Machine Learning Algorithm Selector:**
   - **Standardized Logistic Regression:** Fast, convex, L2-regularized linear baseline with Platt probability calibration.
   - **RBF Support Vector Machine:** Nonlinear kernel classifier capturing complex electrochemical degradation dynamics.
   - **TabPFN 2.5:** Transformer foundation model for tabular classification (`tabpfn-v2-classifier.ckpt`), delivering state-of-the-art out-of-the-box predictions.
5. **Peer Trajectory Centroid Overlay:** Side-by-side visualization of a target battery against its top-5 reference peers and the computed peer centroid trajectory curve $\mu_{\text{peer}}(t)$.
6. **Peer-Match Reliability Index (PMRI):** Interpretable reliability assessment broken down into its four constituent sub-indices: Similarity ($S_{\text{sim}}$), Coverage ($S_{\text{cov}}$), Agreement ($S_{\text{agree}}$), and Consistency ($S_{\text{consist}}$).
7. **Dedicated Algorithm Benchmarking View:** Comprehensive 36-configuration comparative matrix (3 algorithms $\times$ 4 model configs $\times$ 3 cutoffs), dynamic active algorithm cards, AUROC & Brier Score Chart.js comparison charts, and distribution shift analysis.
8. **Research Benchmark:** Master performance table, hypothesis evaluation verdicts (H1–H4), paired bootstrap confidence intervals ($B=1,000$), multi-seed null distribution controls, and selective prediction curves.
9. **Laboratory Hardware Ingestion Protocol:** Technical schema specification and interactive validation interface prepared for external laboratory cycler data ingestion (verifying monotonicity, capacity bounds, and temperature limits).

---

## 2. HeroUI Design System Integration

The dashboard is designed following **HeroUI** design tokens and components:
- **Light Theme (Default):** Clean light backdrop (`#F4F4F5`), crisp white cards (`#FFFFFF`), charcoal/deep navy typography (`#11181C`), muted labels (`#71717A`), and HeroUI Blue primary (`#006FEE`).
- **Dark Theme Toggle:** Seamless toggle switch persisting preference in `localStorage`, using HeroUI dark tokens (`#09090B`, surface `#18181B`, border `#27272A`, text `#ECEDEE`).
- **Semantic Accents:** Success (`#17C964`), Warning (`#F5A524`), Danger (`#F31260`), Secondary Purple (`#7828C8`).
- **Component Primitives:** `.heroui-card`, `.heroui-btn`, `.heroui-chip`, `.heroui-tabs`, `.heroui-table`, `.heroui-progress`, `.heroui-alert`, `.heroui-input`, `.heroui-select`.
- **Persistent Application Shell:** Fixed left sidebar navigation with mobile collapse drawer, sticky top bar with breadcrumbs, global algorithm selector, quick search, and benchmark status badge.

```text
dashboard/
├── assets/
│   └── chart.umd.min.js          # Bundled offline Chart.js v4.4.2 (Zero external CDN dependency)
├── index.html                    # 9-view HeroUI application shell
├── styles.css                    # HeroUI design system stylesheet (Light/Dark themes)
├── app.js                        # Client state manager, algorithm selector, Chart.js renderers
├── server.py                     # Lightweight Python http.server REST API serving real CSV & JSON artifacts
└── README.md                     # Operational documentation and API reference
```

- **Zero Heavy Framework Dependencies:** Built entirely using Python's standard library (`http.server`, `csv`, `json`, `pathlib`) and vanilla ES6+ JavaScript. No Node.js, npm, React, Flask, or Django needed.
- **100% Offline Capable:** Runs without active internet connection; all assets and figures are served locally.
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

## 4. Nine Interactive Navigation Views

1. **Overview (`Battery Health Intelligence`):** Landing page with executive KPI summary (140 total cells, 124 benchmark eligible, 16 excluded channels, 3 cutoffs, 4 models, 372 evaluations), cohort lifecycle degradation chart, leak-free partition status card, and featured batteries table.
2. **Battery Explorer:** Search by Cell ID, multi-criteria filtering by Batch (1, 2, 3), Partition (Train, Primary Test, Excluded), and Risk Class (Elevated vs Normal), with interactive click-to-analyze rows.
3. **Trajectory Analysis:** Deep dive on any individual cell with observation cutoff selection (40, 70, 100 cycles, or Full Lifecycle), multi-sensor telemetry switcher (Discharge Capacity, IR, Max/Avg Temp, Charge Time), factual observations, and grouped 46-feature inspector.
4. **Peer Comparison:** Target battery vs. k=5 reference peers selected strictly from the training cohort; aligned trajectory overlay with mean peer centroid curve, relative deviation table, and PMRI low-reliability caution banner.
5. **EOL Risk Prediction:** Calibrated risk probability gauge ($P(\text{EOL} \le 550)$), decision threshold indicator ($\tau$), risk category badge, comparative 4-model matrix (M1–M4), and active algorithm indicator chip.
6. **Model Evaluation:** Primary test set performance cards (AUROC, AUPRC, Brier Score, ECE, Sensitivity, Specificity), official publication figure viewer (`cutoff_comparison.png`, `model_differences.png`, `pmri_stratification.png`, `calibration_diagrams.png`), and legacy benchmark table.
7. **Algorithm Benchmarking:** Dedicated comparative architecture view featuring the global algorithm selector, 4 active model cards, full 36-configuration comparative benchmark table, AUROC & calibration charts, distribution shift analysis (Batch 3 fast-charging shift), and architecture leaderboard.
8. **PMRI & Confidence:** Deconstructed 4-component PMRI breakdown ($S_{\text{sim}}$, $S_{\text{cov}}$, $S_{\text{agree}}$, $S_{\text{consist}}$), 20-seed empirical null distribution permutation table, and selective prediction risk-coverage trade-off curve.
9. **Hardware Validation:** Laboratory cycler CSV anomaly auditor with drag-and-drop ingestion, built-in valid/corrupted sample loaders, and real-time physical rule checking (cycle monotonicity, non-negative capacity, thermal threshold).

---

## 5. REST API Reference Endpoints

| Endpoint | Method | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/api/overview` | `GET` | None | Returns high-level dataset statistics, model count (4), active algorithm, and risk tallies. |
| `/api/batteries` | `GET` | None | Lists all 140 cells with metadata (batch, policy, lifecycle, split partition, exclusion status). |
| `/api/battery/<id>` | `GET` | `cutoff` (40, 70, 100, all), `algorithm` | Returns cycle history, summary metrics, engineered features, sequestered ground truth, and M1–M4 predictions for the specified algorithm. |
| `/api/algorithm-selector` | `GET` | None | Returns the active algorithm and list of supported algorithms with metadata. |
| `/api/algorithm-selector` | `POST`| Body: `{"algorithm": "..."}` | Sets the global active algorithm across M1–M4 on the server. |
| `/api/algorithm-benchmarks`| `GET`| None | Returns 36-configuration comparative benchmarks, rankings, and distribution shift analysis. |
| `/api/models/predict` | `GET` | `id`, `cutoff`, `model` (M1–M4), `algorithm` | Returns detailed model prediction score, threshold, binary risk classification, PMRI components, and active algorithm. |
| `/api/peers/trajectories`| `GET` | `id`, `cutoff` (40, 70, 100) | Returns aligned trajectories for target cell, top-5 matched peers, mean peer centroid curve, and trajectory deviations. |
| `/api/export/benchmark` | `GET` | `format` (json, csv) | Exports 36-configuration multi-algorithm predictions and metrics in CSV or JSON. |
| `/api/research/algorithms` | `GET` | None | Returns multi-algorithm benchmark metrics across model families. |
| `/api/research/peer-controls` | `GET` | None | Returns 20-seed empirical null permutation test results and PMRI validation analysis. |
| `/api/research/selective-prediction` | `GET` | None | Returns risk-coverage trade-off evaluation records. |
| `/api/hardware/validate`| `POST`| Body: CSV string | Validates external laboratory cycler telemetry CSV against schema and physical bounds rules. |
| `/results/figures/<file>`| `GET` | filename | Serves high-resolution research figures (e.g., `cutoff_comparison.png`, `model_differences.png`). |
