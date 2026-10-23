# Experimental Results & Limitations Report

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Document:** Comparative Benchmark Outcomes, Multi-Algorithm Performance, Selective Prediction, and Research Limitations  
**Execution Date:** 2026-10-22  

---

## 1. Multi-Algorithm Benchmark Summary

Four machine learning algorithm families (Logistic Regression, Random Forest, RBF Support Vector Machine, and Histogram Gradient Boosting) were benchmarked across all four model configurations (M1–M4) and observation cutoffs (40, 70, 100 cycles) on the primary test partition ($N=43$ independent cells).

### Primary Test Discrimination & Calibration Matrix (M4 Configuration)

| Algorithm | Family | Cutoff 40 AUROC | Cutoff 70 AUROC | Cutoff 100 AUROC | Cutoff 40 Brier | Cutoff 70 Brier | Cutoff 100 Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear (Baseline) | 0.8680 | 0.9091 | 0.9065 | 0.1033 | 0.0628 | 0.0815 |
| **Random Forest** | Tree Ensemble | 0.9437 | 0.9697 | 0.9696 | 0.0951 | 0.0646 | 0.0669 |
| **RBF-SVM** | Kernel Method | 0.9351 | 0.9913 | 0.9804 | 0.0831 | 0.0488 | 0.0952 |
| **HistGradientBoosting** | Boosting | 0.9340 | 0.9805 | 0.9674 | 0.0847 | 0.0653 | 0.0735 |

### Critical Empirical Observations
1. **Nonlinear Models Leverage Trajectory & Peer Signal Strongly:**
   - In Random Forest at Cutoff 100, AUROC increases from **0.9217 (M1)** to **0.9696 (M2)** to **0.9761 (M3)**, showing a +0.0544 gain when trajectory and peer features are included. Brier score drops from 0.0901 to 0.0670.
   - In RBF-SVM at Cutoff 70, AUROC peaks at **0.9913** with a Brier score of **0.0488**.
2. **Linear Baseline Resilience:**
   - Standardized Logistic Regression achieves solid discrimination (AUROC 0.8680–0.9329) with well-balanced sensitivity (0.9048–1.0000) and low variance.

---

## 2. Selective Prediction: Risk-Coverage Trade-Off

Using training-only percentile thresholding on trust scores, the system evaluated prognostics accuracy under varying coverage levels on the held-out primary test set at Cutoff 100:

| Target Coverage | Realized Test Coverage | Retained Cells (N) | Retained AUROC | Retained Brier Score | Retained Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **100% (No Abstention)** | 97.7% | 42 / 43 | 0.9108 | 0.0792 | 92.9% |
| **95% Coverage** | 95.3% | 41 / 43 | **0.9372** | **0.0672** | **95.1%** |
| **90% Coverage** | 83.7% | 36 / 43 | 0.9352 | 0.0684 | 94.4% |
| **80% Coverage** | 79.1% | 34 / 43 | 0.9340 | 0.0715 | 94.1% |
| **70% Coverage** | 72.1% | 31 / 43 | 0.9328 | 0.0732 | 93.5% |

*Key Takeaway:* Abstaining on the 2–3 most unreliable predictions (4.7% of the cohort) yields a **+0.0264 increase in AUROC** and reduces classification error from 7.1% to 4.9%.

---

## 3. Scientific Limitations & Boundary Conditions

1. **Cohort Scale:** The benchmark cohort contains 124 eligible cells. While it is one of the largest systematically tested open-access fast-charging battery datasets available, testing across hundreds of distinct cell chemistries (NMC, LCO, sodium-ion) remains an open challenge.
2. **Domain Transfer to Secondary Test (Batch 3):** Performance on Batch 3 shows domain shift due to differing ambient conditions and slightly different channel calibrations. Peer matching successfully identifies that Batch 3 cells have lower PMRI scores, correctly triggering lower confidence warnings.
3. **Software-Only Scope:** This system is an algorithmic screening and prognostics software workstation. It does not interface directly with physical battery hardware, BMS controllers, cell-balancing circuitry, or real-time hardware-in-the-loop (HIL) testbenches.
