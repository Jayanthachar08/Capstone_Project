# PATENT REVISION SUMMARY: TRANSITION FROM DRAFT 1 TO DRAFT 2 REVISED

**Document Revised:** `Patent_DRAFT_1.docx` $\longrightarrow$ `Patent_DRAFT_2_REVISED.docx`  
**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Repository:** `C:\engg notes\7th sem\Capstone_Project`  
**Date of Revision:** October 15, 2026  
**Auditor Roles:** Senior ML Research Engineer, Patent-Technical-Documentation Specialist, and Indian Patent Law Analyst  

---

## 1. Executive Summary of Changes

The technical audit revealed that while `Patent_DRAFT_1.docx` proposed an innovative conceptual architecture, it contained critical omissions in mathematical disclosures, unverified claims of universal predictive superiority, generic illustrative examples, missing drawing numerals, and claim structures vulnerable to Section 3(k) rejection under Indian Patent Law.

`Patent_DRAFT_2_REVISED.docx` comprehensively corrects every technical defect, strictly anchoring every equation, feature count, experimental number, and architectural claim to verified source code and reproducible execution artifacts.

---

## 2. Granular Summary of What Changed and Why

### 2.1 PMRI Formulation & Governing Equations (Critical Correction)
- **What Changed:** Replaced narrative descriptions with exact mathematical decay formulas:
  - Peer Similarity: $S_{\text{similarity}} = \exp(-\bar{d} / d_0)$ with characteristic distance scale $d_0 = 2.0$.
  - Trajectory Agreement: $S_{\text{agreement}} = \exp(-\sigma_{\text{slope}} / \sigma_0)$ with characteristic slope dispersion scale $\sigma_0 = 5 \times 10^{-5}$ Ah/cycle.
  - Availability: $S_{\text{availability}} = \min(1.0, k/5)$ if $k \ge 3$ else $0.0$.
  - Completeness: $S_{\text{completeness}} = N_{\text{valid}} / 7$.
  - Linear Aggregation: $\text{PMRI} = 0.35 S_{\text{similarity}} + 0.30 S_{\text{agreement}} + 0.20 S_{\text{availability}} + 0.15 S_{\text{completeness}}$.
- **Why It Changed:** In Draft 1, the scale constants ($d_0, \sigma_0$) were omitted, rendering the invention mathematically irreproducible and vulnerable to rejection for lack of sufficient disclosure (Section 10(4) of the Indian Patents Act).
- **Conceptual Clarification:** Draft 2 explicitly clarifies that PMRI is an orthogonal measure of peer comparison suitability and reference pool density, **not** a battery failure probability or classifier confidence.

### 2.2 Peer Matching Features, Weights, and Anti-Leakage Rules
- **What Changed:** Formally documented the exact 7 matching features, their physical weighting structure:
  - First fast-charge rate $C_1$ (0.20), SOC switch threshold (0.15), second fast-charge rate $C_2$ (0.15), baseline discharge capacity $Q_d(2)$ (0.15), baseline internal resistance $IR(2)$ (0.15), average temperature $T_{\text{avg}}(2)$ (0.10), cutoff resistance $IR(N_{\text{obs}})$ (0.10).
  - Documented that standardization scalers ($\mu_j, \sigma_j$) are fitted strictly on the 41 training reference cells at each observation cutoff.
  - Formalized the leave-one-out self-exclusion rule ($N_{\text{candidates}} = 40$) for training cells and strict reference pool isolation for test cells.
- **Why It Changed:** Draft 1 vaguely referred to "selected features." Explicit disclosure of weights and anti-leakage rules strengthens non-obviousness and technical credibility.

### 2.3 Feature Dictionary & Model Feature Sets (M1–M4)
- **What Changed:** Embedded an exhaustive 47-row Feature Dictionary (Table 1) detailing names, formulas, engineering units, and input observation windows. Corrected M4's feature membership: M4 includes `pmri`, `peer_similarity_score`, `peer_agreement_score`, and `mean_peer_distance` (not all 4 sub-components).
- **Why It Changed:** Draft 1 cited feature counts (18, 34, 42, 46) without naming the features, and misstated M4's composition.

### 2.4 Experimental Evidence & Honest Scientific Findings
- **What Changed:** Replaced unsupported claims of universal predictive gains with verified quantitative experimental tables (Tables 2 & 3) and paired bootstrap confidence intervals ($B=1000$ resamples):
  - *In-Distribution Parity (`primary_test`, $N=43$):* Baseline M1 is exceptionally strong (AUROC 0.918–0.933); M2, M3, and M4 operate at statistical parity (95% bootstrap CIs for $\Delta\text{AUROC}$ span zero).
  - *Out-of-Distribution Generalization (`secondary_test`, $N=40$ Batch 3):* Under charging policy and structural shift, M1 collapses to AUROC 0.5128 (Cutoff 70), whereas peer-relative models M3 and M4 achieve **AUROC 0.9744** (+0.4616 gain).
  - *Negative Control:* Scrambling peer assignments drops Cutoff 40 AUROC by $-0.0584$ and increases Brier error by $+55\%$.
  - *PMRI Stratification:* High PMRI cells achieve Brier error 0.052 vs 0.074 for Low PMRI cells.
- **Why It Changed:** Draft 1 risked deceptive overstatement. Scientific honesty regarding domain-shift robustness provides a vastly stronger, defensible patent contribution.

### 2.5 Technical Drawings and Reference Numerals
- **What Changed:** Replaced unnumbered overview diagrams with 9 high-resolution (300 DPI) patent figures:
  - FIG. 1: System Architecture (Numerals 100–124)
  - FIG. 2: Operational Flowchart (Numerals 200–222)
  - FIG. 3: Peer Matching Subsystem (Numerals 300–310)
  - FIG. 4: PMRI and Warning Logic (Numerals 400–420)
  - FIG. 5: Model Prediction Architecture (Numerals 500–514)
  - FIG. 6: Cutoff Progression Comparison Plot
  - FIG. 7: Paired Model Differences Forest Plot with 95% Bootstrap CIs
  - FIG. 8: PMRI Stratification Error Analysis Plot
  - FIG. 9: Probability Calibration Reliability Diagrams
- **Why It Changed:** Patent office drafting rules require clear reference numerals cross-referenced in the specification.

### 2.6 Working Embodiment Grounded in Real Data
- **What Changed:** Replaced hypothetical qualitative descriptions in Section 10 with a real-world trace of verified test cell `b1c0` (Cycle 70, protocol 3.6C(80%)-3.6C, peers `b1c1`, `b1c5`, `b1c17`, `b2c29`, `b1c3`, mean dist 0.614, PMRI 0.825, calibrated risk 0.128, NOMINAL RISK, actual life 1190 cycles) contrasted with Batch 3 cell `b3c0`.
- **Why It Changed:** Provides undeniable evidence of reduction to practice.

### 2.7 Claim Restructuring for Indian Patent Law (Section 3(k))
- **What Changed:** Restructured Claims 1–10 into an expanded, legally resilient 12-claim set:
  - Claim 1 (Independent System): Anchored in physical electrochemical sensor hardware, physical cycling signals, domain-shift normalization, and controller output signals (protective derating or advisory alert).
  - Claim 9 (Independent Method): Formulated as a technical method for operating a battery degradation monitoring system with physical sensor inputs and control outputs.
  - Claims 2–8 & 10–12 (Dependent): Reciting exact 7-feature weights, 8 peer-relative features, exponential decay PMRI equations, progressive M1–M4 hierarchy, anti-leakage isolation, and BMS hardware integration.
- **Why It Changed:** Protects against summary rejection under Section 3(k) by demonstrating a concrete "technical effect" under the Delhi High Court's *Ferid Allani* doctrine.

---

## 3. What Remains Unresolved / Requires Author Confirmation

1. **Intended Commercial Integration:**  
   The claims and description recite integration with battery management systems (BMS), manufacturing cell-sorting lines, and test-bench controllers. The author should confirm whether the primary commercial target is laboratory test-bench software or onboard EV battery management firmware.
2. **Alternative Chemistry Expansion:**  
   The experimental embodiment is validated exclusively on commercial A123 1.1 Ah LFP cylindrical cells. If experimental data for NMC or pouch cells becomes available in future work, it can be filed as a continuation or divisional application.

---

## 4. What Requires Review by a Registered Patent Agent / Attorney

1. **Formal Claim Scope and Language:**  
   While the revised claims are drafted to maximize technical effect and overcome Section 3(k), the final claim phrasing should be vetted by an Indian registered patent attorney to align with current Indian Patent Office (IPO) examination trends.
2. **Jurisdiction-Specific National Phase Filings:**  
   If international protection (PCT or direct foreign filings in USPTO/EPO) is contemplated, claims may be formatted with broader apparatus and system language tailored to US 35 U.S.C. § 101 / Alice guidelines or EPO EPC Article 52(2)/(3) technical character standards.
3. **Patentability Disclaimer:**  
   No audit or revision can guarantee patent grant. Patentability depends on the official prior-art search and final examination by the Controller of Patents.
