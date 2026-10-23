# Capstone Review 1 Preparation Summary

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Degree:** B.Tech Computer Science & Engineering (Capstone Project)  
**Document:** Comprehensive Defense Briefing for Review 1 Committee  
**Date:** 2026-10-22  

---

## 1. Project Overview & Research Problem

- **Core Problem:** Predicting battery cycle life from early operational data (first 40–100 cycles) is notoriously difficult because standard degradation trajectories exhibit flat early capacity curves with minimal observable fade.
- **Novel Scientific Premise:** An individual battery's early degradation is best interpreted not in absolute isolation, but relative to a geometrically matched peer group of reference cells operating under identical initial conditions.
- **Confidence-Aware Mechanism:** Peer-relative degradation features are paired with an explicit Peer-Match Reliability Index (PMRI) and counterfactual peer instability penalties ($U$), enabling the system to identify unreliable comparisons and selectively abstain from uncertain predictions.

---

## 2. Key Accomplishments for Review 1

### A. Data Engineering & Integrity
- Parsed 140 raw commercial A123 18650 LFP cells from the MATR/Severson Nature Energy benchmark.
- Enforced documented exclusion rules (16 excluded, 124 benchmark-eligible cells).
- Structured 372 cutoff instances across $N_{\text{obs}} \in \{40, 70, 100\}$ cycles.
- Formulated primary binary target `target_fail_within_500` ($500$-cycle post-cutoff failure horizon; balanced $\sim 51\%$ class distribution).

### B. Feature Engineering & Multi-Algorithm Models
- Engineered 46 features across 6 functional groups (11 initial condition, 7 snapshot, 11 trajectory, 5 differential curves, 8 peer-relative, 4 peer reliability).
- Implemented and evaluated four model configurations (M1: 18, M2: 34, M3: 42, M4: 46 features).
- Benchmarked 4 machine learning algorithm families: Standardized Logistic Regression, Random Forest, RBF Support Vector Machine, and Histogram Gradient Boosting.
- Developed M5 Adaptive Peer-Trust Engine combining PMRI and counterfactual peer perturbation sensitivity ($U$).

### C. Rigorous Validation & Anti-Leakage
- Strictly disjoint battery-level splits: Train (41), Primary Test (43), Secondary Test (40).
- Zero post-cutoff temporal leakage: verified through regression tests.
- 100% training-only threshold tuning (Youden's J) and probability calibration (Platt scaling).
- Test cells select reference peers exclusively from the training reference pool.

### D. Empirical Proofs & Statistical Significance
- Multi-seed negative control (20 distinct seeds): Shuffling peer associations at Cutoff 40 drops AUROC from **0.8723 to 0.8038 &plusmn; 0.0344** ($p = 0.0000$), demonstrating non-random peer predictive value.
- PMRI Error Stratification: Batteries with $\text{PMRI} \ge \text{median}$ exhibit **3.2× to 33× lower Brier loss** than low PMRI cells across all cutoffs.
- Selective Prediction: Abstaining on the 4.7% most uncertain predictions at Cutoff 100 increases test AUROC from **0.9108 to 0.9372** and accuracy to **95.1%**.

### E. Professional Software Engineering & Dashboard
- Native Python standard library HTTP server (`dashboard/server.py`) with zero external web framework dependencies.
- 9 interactive dashboard views with real-time Chart.js trajectory overlays, peer matching diagnostics, M5 waterfall sensitivity, multi-algorithm comparisons, and laboratory cycler CSV hardware validation.
- **83 automated unit tests passing 100%** in ~1.8 seconds.

---

## 3. Quick Reference Metric Card (Primary Test Cohort, N = 43)

| Model Configuration | Best Algorithm | Cutoff 40 AUROC | Cutoff 70 AUROC | Cutoff 100 AUROC | Cutoff 100 Brier |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Baseline Snapshot** | RBF-SVM | 0.9524 | 0.9805 | 0.9522 | 0.0659 |
| **M2: Individual Trajectory** | RBF-SVM | 0.9459 | 0.9978 | 0.9783 | 0.0864 |
| **M3: Peer-Relative** | RBF-SVM | 0.9416 | 0.9935 | **0.9804** | 0.0934 |
| **M4: Confidence-Aware Peer** | Random Forest | **0.9437** | 0.9697 | 0.9696 | **0.0669** |
| **M5: Adaptive Peer-Trust** | Dynamic Blend | 0.8874 | 0.9156 | 0.9108 | **0.0792** |
