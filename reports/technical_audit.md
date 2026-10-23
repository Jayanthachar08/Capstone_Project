# Technical Baseline Audit & Risk Assessment Report

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Repository Location:** `C:\engg notes\7th sem\Capstone_Project`  
**Audit Date:** 2026-10-22  
**Auditor:** Senior ML Research Engineer & Statistical Validation Auditor  
**Operating Environment:** Windows 11, PowerShell, Python 3.11.9 (`.venv`)  

---

## 1. Executive Summary & Verification of Baseline

An exhaustive audit of the repository, source code, data pipelines, model configurations, test suite, and dashboard was conducted prior to modifying code or launching new experiments.

- **Baseline Test Suite:** Executed `python -m unittest discover -s tests -v`. **79 out of 79 unit tests passed** in 2.37 seconds with zero failures and zero errors.
- **Repository Integrity:** Clean Git working tree on branch `main` (`origin/main`). Existing research outputs in `results/`, `docs/`, and `data/` were preserved intact.
- **Dataset Audit:**
  - Total raw cells: 140 (Batches 1, 2, and 3 from the MATR/Severson et al. benchmark).
  - Excluded cells: 16 cells (excluded according to documented Severson criteria for channel issues, early cycle termination < 100 cycles, or missing cycles).
  - Benchmark-eligible cells: 124 cells.
  - Cutoff records: 372 rows (124 eligible cells × 3 cutoffs at $N_{\text{obs}} \in \{40, 70, 100\}$ cycles).
- **Split Assignments:**
  - Train: 41 cells (123 cutoff observations)
  - Primary Test: 43 cells (129 cutoff observations; shared batch domain)
  - Secondary Test: 40 cells (120 cutoff observations; Batch 3 domain)
  - Cell-level disjointness: 100% verified. Every cycle observation of any cell is strictly isolated to its assigned split.

---

## 2. Path Traceability: Ingestion to Dashboard

```
data/raw/ (MATR HDF5 batches)
  │
  ▼
data/processed/metadata.csv & cycle_summaries.csv
  │
  ├──► src/features/feature_extractor.py ──► features_snapshot.csv, features_trajectory.csv, features_curves.csv
  │
  ├──► src/data/target_builder.py ─────────► targets.csv (EOL at 80% nominal; target_fail_within_500)
  │
  ▼
data/processed/modeling_dataset.csv (34 individual features + target + split metadata)
  │
  ├──► src/peers/peer_matcher.py ──────────► k=5 reference peers from training pool only
  ├──► src/peers/peer_reliability.py ──────► PMRI composite score [0, 1]
  │
  ▼
data/processed/peer_features.csv (46 features total: M1=18, M2=34, M3=42, M4=46)
  │
  ├──► src/models/train_models.py ─────────► results/phase5_model_metrics.csv, phase5_predictions.csv
  ├──► src/evaluation/hypothesis_evaluator ─► results/phase6_hypothesis_summary.json
  ├──► src/models/evaluate_m5.py ──────────► results/m5_predictions.csv, m5_peer_robustness.csv
  │
  ▼
dashboard/server.py ──► REST API endpoints ──► dashboard/app.js (9 Interactive UI Views)
```

---

## 3. Confirmed Implementation Bugs & Code Defects

### Defect 1: Arbitrary Sentinel Value `-10.0` in Differential Curve Extraction
- **Location:** `src/features/feature_engineering.py:127`
- **Code:**
  ```python
  log10_var = float(np.log10(var_val)) if var_val > 0 else -10.0
  ```
- **Analysis:** If $\text{var}(\Delta Q(V)) \le 0$, the mathematical $\log_{10}$ is undefined ($-\infty$). Substituting an arbitrary constant `-10.0` is dangerous: when passed through downstream `StandardScaler` or linear models, `-10.0` is treated as a valid numerical observation located $\sim 4$ standard deviations below the mean, which distorts regression weights.
- **Dataset Impact Check:** In `data/processed/modeling_dataset.csv`, the actual observed $\log_{10}(\text{var})$ values range between $-5.77$ and $-1.02$. There are zero `-10.0` sentinel occurrences in the processed dataset.
- **Correction Required:** If `var_val <= 0`, assign `np.nan` and handle missingness through documented, training-only imputation rather than injecting magic numeric sentinels.

---

## 4. Methodological & Statistical Risks

### Risk 1: High Predictive Separation of Baseline Initial Conditions (M1)
- **Observation:** M1 achieves an AUROC of 0.868–0.933 on the primary test set using only 18 features (charging policy $C_1, \text{SOC}, C_2$ plus cycle 2 capacity/impedance and cycle $N_{\text{obs}}$ snapshot).
- **Risk:** Because baseline discrimination is already exceptionally strong, marginal AUROC gains from M2, M3, and M4 are modest ($\Delta \text{AUROC} \approx \pm 0.01$).
- **Mitigation:** Model superiority must not be claimed solely on AUROC. Scientific evaluation must prioritize Brier score, Expected Calibration Error (ECE), threshold stability, sensitivity at operational operating points, and counterfactual robustness.

### Risk 2: Single-Algorithm Bias (Phase 5 evaluated only Logistic Regression)
- **Observation:** Phase 5 implemented only standardized Logistic Regression. While Logistic Regression is transparent and leak-resistant, tree ensembles and kernel methods handle nonlinear feature interactions differently.
- **Mitigation:** Implement a modular multi-algorithm benchmark comparing Logistic Regression, Random Forest, RBF-SVM, and Histogram Gradient Boosting with identical training-only CV threshold and calibration protocols.

### Risk 3: Single-Seed Randomized Peer Control
- **Observation:** Phase 5 verified negative control using only `random_state=42`. A single seed could produce an uncharacteristically high or low null result.
- **Mitigation:** Implement a multi-seed (20 distinct seeds) randomized peer control to construct empirical 95% confidence intervals for the null distribution.

### Risk 4: Selective Prediction and Abstention Leakage
- **Observation:** Abstaining on low-PMRI or high-uncertainty batteries can improve held-out accuracy, but if the abstention threshold is selected on test data, results are over-optimistic.
- **Mitigation:** All abstention thresholds must be chosen strictly on 5-fold training out-of-fold validation sets, with held-out coverage and error evaluated on the test set.

---

## 5. Dashboard Audit

- **Technology Stack:** Native Python standard library `http.server` backend (`dashboard/server.py`) serving static assets and dynamic JSON REST endpoints. Frontend built with vanilla JavaScript, modern CSS custom properties, and local/CDN Chart.js.
- **Views:** 9 active views (`overview`, `explorer`, `trajectories`, `peers`, `m5`, `risk`, `benchmark`, `hardware`, `about`).
- **Data Source:** Zero hardcoded synthetic numbers in API handlers; all data is dynamically loaded from processed CSVs and JSONs.
- **Enhancement Target:** Expand the Research Benchmark and Experiments views to support multi-algorithm side-by-side comparisons, multi-seed peer control distributions, and selective prediction coverage curves.

---

## 6. Action Plan & Progression Gates

1. **Gate 1:** Fix sentinel `-10.0` in `feature_engineering.py` and add post-cutoff invariance tests.
2. **Gate 2:** Implement modular multi-algorithm interface in `src/models/` (LR, RF, RBF-SVM, HistGBM).
3. **Gate 3:** Execute multi-algorithm benchmark and multi-seed randomized peer controls; save versioned artifacts.
4. **Gate 4:** Implement training-only selective prediction and author `docs/DECISION_POLICY.md`.
5. **Gate 5:** Expose multi-algorithm and selective prediction data in dashboard API and UI.
6. **Gate 6:** Run full test suite, verify 100% pass, and deliver comprehensive reports.
