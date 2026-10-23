# Patentability & Prior Disclosure Assessment Report

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Degree:** B.Tech Computer Science & Engineering (Capstone Project)  
**Author:** Jayanth Achar  
**Repository:** `https://github.com/Jayanthachar08/Capstone_Project`  
**Date:** October 2026  

---

## 1. Executive Summary

This report evaluates the intellectual property (IP) status, novelty considerations, and patentability of the algorithmic innovations developed in this Capstone Project, specifically:
1. **The Peer-Match Reliability Index (PMRI):** A multi-criteria confidence index quantifying reference cohort density, trajectory agreement, and candidate pool coverage.
2. **Counterfactual Peer Perturbation Instability ($U$):** A systematic sensitivity score derived from Leave-One-Peer-Out (LOO), Single-Peer Replacement, and Leave-Two-Out perturbations.
3. **Adaptive Peer-Trust Scaling ($T = R \cdot \exp(-\lambda U)$):** An exponential decay formulation weighting peer-relative versus individual degradation history models.
4. **Calibrated Blended Risk ($p_{M5}$) and Selective Abstention:** A dual-engine prognostic framework with formal uncertainty gating.

---

## 2. Technical Novelty Analysis

### 2.1 Contrast with Prior Art
- **Standard Baseline (Severson et al., Nature Energy 2019):** Uses early cycle differences ($\Delta Q_{100-10}(V)$) in an unconstrained global elastic net or random forest regression. Operates on cells in isolation without reference cohort context.
- **k-Nearest Neighbor Trajectory Retrieval (e.g. Wang et al., Hu et al.):** Classic instance-based approaches assign equal weight to retrieved neighbors and do not evaluate the counterfactual stability of the neighbor set.
- **Proposed M5 Framework:** 
  - Generates 12 systematic counterfactual neighbor configurations per query cell.
  - Explicitly measures prediction variance across counterfactual neighbor sets ($U$).
  - Dynamically dials down trust in peer-relative features when instability is elevated, falling back safely to the battery's individual trajectory history.
  - Triggers explicit selective abstention when reference pool coverage is insufficient or counterfactual volatility is excessive.

---

## 3. Prior Public Disclosure Audit

A thorough audit of public disclosures associated with this project reveals the following timeline:

1. **University Capstone Submissions & Progress Evaluations:** Intermediate project presentations, reports, and code reviews submitted to the Department of Computer Science & Engineering.
2. **Public GitHub Repository:** The complete codebase, commit history, feature definitions, experimental results, and dashboard server have been hosted publicly on GitHub at:
   ```text
   https://github.com/Jayanthachar08/Capstone_Project
   ```
3. **Public Technical Reports:** Phase 4, Phase 5, Phase 6, and Phase 7 evaluation reports documenting the exact mathematical formulations, hyperparameters ($\lambda$), decision thresholds ($\tau$), and ablation tables.

---

## 4. Legal Implications under Patent Law

### 4.1 International Jurisdictions (Absolute Novelty Standard)
- **European Patent Office (EPC Article 54):** The European Patent Convention adheres to a strict "absolute novelty" standard. Any disclosure of the invention made available to the public anywhere in the world by means of a written description, oral presentation, public repository, or use prior to the filing date permanently destroys patent novelty.
- **State Intellectual Property Office (China, CNIPA) & Japan Patent Office (JPO):** Strict novelty requirements apply with narrow exceptions only for government-approved academic conferences.
- **Conclusion:** Public availability of the GitHub repository prior to a patent application filing constitutes an absolute bar to patent protection across Europe and most international jurisdictions.

### 4.2 United States (35 U.S.C. § 102)
- **Grace Period:** Under 35 U.S.C. § 102(b)(1), disclosures made by the inventor within one year prior to the effective filing date do not qualify as prior art against the inventor (the "one-year personal grace period").
- **Application:** If the initial public commit or repository publication occurred within the preceding 12 months, patent eligibility in the United States may theoretically remain open under this grace period—provided a provisional patent application is filed prior to the 12-month expiration date.
- **Statutory Subject Matter (35 U.S.C. § 101):** Under the *Alice/Mayo* framework, mathematical algorithms, statistical formulas (such as $T = R \cdot \exp(-\lambda U)$), and software abstract ideas are non-patentable unless tied directly to a practical, non-conventional technological transformation (e.g. an embedded Battery Management System hardware controller with physical switching actions).

---

## 5. Strategic IP Recommendations

1. **Academic Publication & Open-Source Portfolio:**
   - The primary value of this undergraduate capstone project is academic rigor, reproducibility, and demonstrable machine learning engineering competence.
   - The project is ideally suited for peer-reviewed journal submission (e.g., *IEEE Transactions on Industrial Informatics*, *Journal of Energy Storage*, or *Applied Energy*) or open-access preprint publication on arXiv.
2. **Software Copyright & Permissive Licensing:**
   - Software implementation is automatically protected by international copyright law upon creation.
   - Recommended license: **MIT License** or **Apache 2.0**, encouraging widespread scientific citation while reserving creator attribution.
3. **Defensive Publication:**
   - By publishing the methodology, PMRI formulations, and counterfactual perturbation engine openly in the public domain, this project functions as a **defensive publication**—preventing third-party commercial entities from claiming exclusive patent rights over the same methodology in the future.
