# Multi-Algorithm Machine Learning Benchmark Report

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Validation Protocol:** 5-Fold Stratified Training CV (Threshold & Calibration Tuning strictly Training-Only)  
**Primary Test Partition:** N = 43 independent cells (shared batch domain)  
**Target Horizon:** `target_fail_within_500` (Failure within 500 cycles post-cutoff)  

---

## 1. Primary Test Partition Comparison (AUROC, AUPRC, Brier Score)

| Cutoff | Algorithm | Family | Model | AUROC | AUPRC | Brier Score | Sensitivity | Specificity | F1-Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **40** | `hist_gradient_boosting` | Boosting | **M1** | 0.9286 | 0.8612 | 0.0842 | 0.5714 | 0.9545 | 0.7059 |
| **40** | `hist_gradient_boosting` | Boosting | **M2** | 0.9340 | 0.8569 | 0.0847 | 0.6667 | 0.9545 | 0.7778 |
| **40** | `hist_gradient_boosting` | Boosting | **M3** | 0.9340 | 0.8569 | 0.0847 | 0.6667 | 0.9545 | 0.7778 |
| **40** | `hist_gradient_boosting` | Boosting | **M4** | 0.9340 | 0.8569 | 0.0847 | 0.6667 | 0.9545 | 0.7778 |
| **40** | `logistic_regression` | Linear | **M1** | 0.9177 | 0.8461 | 0.0764 | 0.9524 | 0.8636 | 0.9091 |
| **40** | `logistic_regression` | Linear | **M2** | 0.9091 | 0.8357 | 0.0871 | 0.9048 | 0.9091 | 0.9048 |
| **40** | `logistic_regression` | Linear | **M3** | 0.8723 | 0.7834 | 0.1015 | 0.9048 | 0.8636 | 0.8837 |
| **40** | `logistic_regression` | Linear | **M4** | 0.8680 | 0.7437 | 0.1033 | 0.9048 | 0.8182 | 0.8636 |
| **40** | `random_forest` | Tree Ensemble | **M1** | 0.9307 | 0.8704 | 0.0938 | 0.9524 | 0.8636 | 0.9091 |
| **40** | `random_forest` | Tree Ensemble | **M2** | 0.9416 | 0.8607 | 0.0958 | 0.9048 | 0.9091 | 0.9048 |
| **40** | `random_forest` | Tree Ensemble | **M3** | 0.9589 | 0.9216 | 0.0806 | 1.0000 | 0.8636 | 0.9333 |
| **40** | `random_forest` | Tree Ensemble | **M4** | 0.9437 | 0.8625 | 0.0951 | 0.9048 | 0.9091 | 0.9048 |
| **40** | `rbf_svm` | Kernel Method | **M1** | 0.9524 | 0.9246 | 0.0717 | 0.9524 | 0.9091 | 0.9302 |
| **40** | `rbf_svm` | Kernel Method | **M2** | 0.9459 | 0.8644 | 0.0690 | 0.9048 | 0.9091 | 0.9048 |
| **40** | `rbf_svm` | Kernel Method | **M3** | 0.9416 | 0.8607 | 0.0759 | 0.9524 | 0.8182 | 0.8889 |
| **40** | `rbf_svm` | Kernel Method | **M4** | 0.9351 | 0.8534 | 0.0831 | 0.9524 | 0.8182 | 0.8889 |
| **70** | `hist_gradient_boosting` | Boosting | **M1** | 0.9729 | 0.9738 | 0.0628 | 0.5000 | 1.0000 | 0.6667 |
| **70** | `hist_gradient_boosting` | Boosting | **M2** | 0.9805 | 0.9829 | 0.0653 | 0.5455 | 1.0000 | 0.7059 |
| **70** | `hist_gradient_boosting` | Boosting | **M3** | 0.9805 | 0.9829 | 0.0653 | 0.5455 | 1.0000 | 0.7059 |
| **70** | `hist_gradient_boosting` | Boosting | **M4** | 0.9805 | 0.9829 | 0.0653 | 0.5455 | 1.0000 | 0.7059 |
| **70** | `logistic_regression` | Linear | **M1** | 0.9329 | 0.9106 | 0.0647 | 1.0000 | 0.8571 | 0.9362 |
| **70** | `logistic_regression` | Linear | **M2** | 0.9156 | 0.8265 | 0.0620 | 1.0000 | 0.8571 | 0.9362 |
| **70** | `logistic_regression` | Linear | **M3** | 0.9091 | 0.8082 | 0.0620 | 1.0000 | 0.8571 | 0.9362 |
| **70** | `logistic_regression` | Linear | **M4** | 0.9091 | 0.8082 | 0.0628 | 1.0000 | 0.8571 | 0.9362 |
| **70** | `random_forest` | Tree Ensemble | **M1** | 0.9773 | 0.9766 | 0.0641 | 0.9545 | 0.9048 | 0.9333 |
| **70** | `random_forest` | Tree Ensemble | **M2** | 0.9848 | 0.9839 | 0.0588 | 1.0000 | 0.9524 | 0.9778 |
| **70** | `random_forest` | Tree Ensemble | **M3** | 0.9784 | 0.9748 | 0.0580 | 1.0000 | 0.9524 | 0.9778 |
| **70** | `random_forest` | Tree Ensemble | **M4** | 0.9697 | 0.9588 | 0.0646 | 0.8636 | 0.9524 | 0.9048 |
| **70** | `rbf_svm` | Kernel Method | **M1** | 0.9805 | 0.9809 | 0.0518 | 1.0000 | 0.8571 | 0.9362 |
| **70** | `rbf_svm` | Kernel Method | **M2** | 0.9978 | 0.9980 | 0.0387 | 0.9545 | 0.9524 | 0.9545 |
| **70** | `rbf_svm` | Kernel Method | **M3** | 0.9935 | 0.9941 | 0.0453 | 0.9545 | 0.9048 | 0.9333 |
| **70** | `rbf_svm` | Kernel Method | **M4** | 0.9913 | 0.9920 | 0.0488 | 1.0000 | 0.8095 | 0.9167 |
| **100** | `hist_gradient_boosting` | Boosting | **M1** | 0.9511 | 0.9607 | 0.0797 | 0.4783 | 1.0000 | 0.6471 |
| **100** | `hist_gradient_boosting` | Boosting | **M2** | 0.9674 | 0.9769 | 0.0735 | 0.5217 | 1.0000 | 0.6857 |
| **100** | `hist_gradient_boosting` | Boosting | **M3** | 0.9674 | 0.9769 | 0.0735 | 0.5217 | 1.0000 | 0.6857 |
| **100** | `hist_gradient_boosting` | Boosting | **M4** | 0.9674 | 0.9769 | 0.0735 | 0.5217 | 1.0000 | 0.6857 |
| **100** | `logistic_regression` | Linear | **M1** | 0.9217 | 0.9114 | 0.0859 | 0.9565 | 0.8500 | 0.9167 |
| **100** | `logistic_regression` | Linear | **M2** | 0.9152 | 0.8477 | 0.0807 | 0.9565 | 0.8500 | 0.9167 |
| **100** | `logistic_regression` | Linear | **M3** | 0.9087 | 0.8378 | 0.0801 | 0.9565 | 0.8500 | 0.9167 |
| **100** | `logistic_regression` | Linear | **M4** | 0.9065 | 0.8276 | 0.0815 | 0.9565 | 0.8500 | 0.9167 |
| **100** | `random_forest` | Tree Ensemble | **M1** | 0.9217 | 0.9445 | 0.0901 | 0.9565 | 0.8500 | 0.9167 |
| **100** | `random_forest` | Tree Ensemble | **M2** | 0.9696 | 0.9760 | 0.0716 | 0.9130 | 0.9500 | 0.9333 |
| **100** | `random_forest` | Tree Ensemble | **M3** | 0.9761 | 0.9782 | 0.0670 | 0.9565 | 0.8500 | 0.9167 |
| **100** | `random_forest` | Tree Ensemble | **M4** | 0.9696 | 0.9695 | 0.0669 | 0.9565 | 0.9500 | 0.9565 |
| **100** | `rbf_svm` | Kernel Method | **M1** | 0.9522 | 0.9711 | 0.0659 | 0.9565 | 0.9000 | 0.9362 |
| **100** | `rbf_svm` | Kernel Method | **M2** | 0.9783 | 0.9868 | 0.0864 | 0.9565 | 0.8000 | 0.8980 |
| **100** | `rbf_svm` | Kernel Method | **M3** | 0.9804 | 0.9878 | 0.0934 | 0.9565 | 0.8000 | 0.8980 |
| **100** | `rbf_svm` | Kernel Method | **M4** | 0.9804 | 0.9869 | 0.0952 | 0.9565 | 0.8000 | 0.8980 |

---

## 2. Algorithm Summary Across Cutoffs (M4 Configuration)

| Algorithm | Cutoff 40 AUROC | Cutoff 70 AUROC | Cutoff 100 AUROC | Cutoff 40 Brier | Cutoff 70 Brier | Cutoff 100 Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_regression` | 0.8680 | 0.9091 | 0.9065 | 0.1033 | 0.0628 | 0.0815 |
| `random_forest` | 0.9437 | 0.9697 | 0.9696 | 0.0951 | 0.0646 | 0.0669 |
| `rbf_svm` | 0.9351 | 0.9913 | 0.9804 | 0.0831 | 0.0488 | 0.0952 |
| `hist_gradient_boosting` | 0.9340 | 0.9805 | 0.9674 | 0.0847 | 0.0653 | 0.0735 |