# COMPREHENSIVE TECHNICAL AUDIT REPORT: PATENT DRAFT 1

**Document Audited:** `Patent_DRAFT_1.docx` (16 pages, reference draft)  
**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Repository:** `C:\engg notes\7th sem\Capstone_Project`  
**Auditor Roles:** Senior ML Research Engineer, Patent-Technical-Documentation Specialist, Software Auditor, and Prior-Art Research Assistant  
**Date of Audit:** October 15, 2026  
**Audit Standard:** Zero hallucination. Every statement, formula, numerical threshold, and claim must be traceable to verified source code, datasets, test suites, or execution artifacts.

---

## 1. Executive Summary of Audit Findings

The reference draft `Patent_DRAFT_1.docx` establishes a valuable architectural framework combining similarity-based peer selection, peer-relative degradation features, a composite reliability index (PMRI), and machine-learning risk screening. However, a rigorous audit against the underlying codebase (`src/`), processed datasets (`data/processed/`), test suites (`tests/`), and execution results (`results/`) reveals **critical technical defects, numerical discrepancies, incomplete formula disclosures, and over-generalized claims** that would jeopardize technical credibility and patentability if left uncorrected.

### Summary of Key Findings:

1. **PMRI Formula & Component Under-Disclosure (Critical Technical Defect):**  
   The draft states the composite linear weighting of PMRI ($0.35 S_{\text{similarity}} + 0.30 S_{\text{agreement}} + 0.20 S_{\text{availability}} + 0.15 S_{\text{completeness}}$) and notes warning thresholds, but **completely omits the exact mathematical decay formulas and scale parameters** ($d_0 = 2.0$, $\sigma_0 = 5 \times 10^{-5}$ Ah/cycle) required to compute $S_{\text{similarity}}$ and $S_{\text{agreement}}$. A person skilled in the art cannot reproduce PMRI without these non-linear governing equations. Furthermore, the draft conflates PMRI in certain passages with a failure probability or calibrated prediction confidence.
   
2. **Peer Matching Feature Definitions & Discrepancies:**  
   The draft mentions weighted Euclidean distance over "selected matching features," but does not explicitly document the exact 7 matching features, their exact physical weights ($0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10$), or the fact that standardization parameters ($\mu, \sigma$) are fitted strictly on the 41 training reference cells at each observation cutoff.

3. **M4 Feature Set Membership Inexactness:**  
   The draft states that M4 contains 46 features including "four PMRI component features." In the actual implementation (`src/models/model_factory.py`), the four added features are `pmri` (composite), `peer_similarity_score`, `peer_agreement_score`, and `mean_peer_distance`—not the four sub-components ($S_{\text{avail}}$ and $S_{\text{comp}}$ are not individual features in M4; `mean_peer_distance` is used instead).

4. **Experimental Population & Partition Numbers:**  
   The draft lacks exact cell counts. The actual experimental population comprises **140 raw battery cells**, of which **124 are benchmark-eligible** and **16 are excluded** based on objective physical criteria. The eligible cohort is partitioned strictly into **41 training cells**, **43 primary test cells** (in-distribution), and **40 secondary test cells** (out-of-distribution Batch 3 cells with modified structural packaging).

5. **Honest Scientific Evaluation of Experimental Evidence (M1–M4):**  
   The draft implies that peer-relative features (M3) and PMRI (M4) universally improve discrimination metrics over baselines. The actual verified results (`results/phase5_model_metrics.csv` and `results/phase6_model_differences.csv`) reveal:
   - **In-distribution (`primary_test`, $N=43$):** Baseline M1 is exceptionally strong (AUROC 0.9177–0.9329) because initial charging protocols and impedance dominate early life in the Severson dataset. Adding individual trajectory (M2), peer-relative features (M3), and PMRI (M4) achieves **statistical parity** (AUROC 0.8680–0.9091; all 95% bootstrap confidence intervals for $\Delta\text{AUROC}$ span zero).
   - **Out-of-distribution (`secondary_test`, $N=40$ Batch 3):** Under structural/protocol domain shift, baseline M1 suffers severe collapse (AUROC 0.5128 at Cutoff 70), whereas peer-relative models M3 and M4 maintain **AUROC 0.9744** (+0.4616 over baseline), demonstrating that peer-relative features provide crucial domain normalization across operating policy shifts.
   - **PMRI Role:** PMRI as an additive linear feature does not improve AUROC ($\Delta\text{AUROC} \approx 0.00$), but as a **confidence gate / meta-reliability filter**, it reliably isolates low-error predictions (Brier score 0.052 for High PMRI vs 0.074 for Low PMRI) and prevents overconfident false alarms.
   
6. **Patent Scope & India Patent Law Compliance (Section 3(k)):**  
   Claims 1–10 in Draft 1 describe software modules and algorithms in terms that risk rejection under Section 3(k) of the Indian Patents Act, 1970 ("computer programme per se or algorithms"). The claims must be restructured to emphasize the **concrete physical-technical system**, cycle-indexed electrochemical sensor signals, physical reference pools, domain-shift compensation, and hardware-in-the-loop operational safety interventions (advisory abstention / derating control), aligning with the Delhi High Court's *Ferid Allani* standard of "technical effect".

---

## 2. Document Audit Table: Section-by-Section Analysis

| Item # | Draft Page / Section | Current Draft Statement | Issue / Uncertainty | Verified Implementation Evidence | Required Document Correction | Error Severity |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| **A1** | Title & Header | *Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries* | Accurately reflects project research, but lacks specific reference to operational screening / hardware interface in patent claims. | Repo Title in `README.md` and codebase. | Retain title; align patent description with concrete physical battery measurement systems. | Minor / Verification |
| **A2** | Section 1: Field of Invention | Describes general computer-implemented system and method for early EOL risk screening. | Framed purely as data processing; needs stronger anchoring to battery management systems, electrochemical sensor telemetry, and physical cell degradation. | `src/features/feature_engineering.py` extracts physical electrochemical features (voltage curves, charge times, impedance). | Emphasize physical transformation: converting electrochemical cycling measurements into health indicators and protective control signals. | Moderate |
| **A3** | Section 3: Objectives of Invention | Objective 4 states: "To develop a Peer-Match Reliability Index (PMRI) that combines peer similarity, degradation-trajectory agreement, peer availability, and feature completeness..." | Does not define the mathematical operationalization, non-linear decay kernels, or characteristic scales. | `src/peers/peer_reliability.py` lines 18–32, 69–102. | Provide full mathematical disclosure of exponential decay functions, scale constants, and boundary behavior. | Critical |
| **A4** | Section 5: Summary of Invention | States peer matching uses "selected operating-condition and battery-condition features... weighted Euclidean distance... select up to five nearest reference cells... minimum of three valid peers." | Does not specify the exact 7 features or their physical weighting rationale (50% charging regime, 35% initial health, 15% cutoff impedance). | `src/peers/peer_matcher.py` lines 20–39 (`DEFAULT_MATCHING_FEATURES`, `DEFAULT_FEATURE_WEIGHTS`). | Explicitly list all 7 features, their normalized weights, and physical significance. | Substantive |
| **A5** | Section 5: Summary of Invention | States PMRI combines four components with configurable weights (0.35, 0.30, 0.20, 0.15). | Omits scale parameters $d_0=2.0$ and $\sigma_0=5\times 10^{-5}$ Ah/cycle, leaving the calculation irreproducible. | `src/peers/peer_reliability.py` lines 27–28. | Incorporate complete equations for $S_{\text{sim}}$, $S_{\text{agree}}$, $S_{\text{avail}}$, and $S_{\text{comp}}$. | Critical |
| **A6** | Section 5 & 7: EOL Definition & Target | Defines EOL as discharge capacity reaching 80% of nominal capacity, target as EOL within 500 cycles after cutoff. | Does not specify nominal capacity (1.1 Ah), threshold value (0.88 Ah), or cycle 1 diagnostic exclusion rule. | `src/features/target_builder.py` lines 18–26; `src/features/feature_engineering.py` lines 168–170. | Clarify nominal cell rating (1.1 Ah A123 Systems APR18650M1A LFP/graphite), 0.88 Ah cutoff, and cycle 2 physical baselining. | Substantive |
| **A7** | Section 6: Drawings (FIG. 1 & FIG. 2) | FIG. 1 and FIG. 2 embedded in draft are overview flowcharts without standard patent reference numerals (e.g., 100, 102, 104). | Patent drawings must have consistent numerical callouts cross-referenced in the Detailed Description. | Two diagrams extracted from draft docx. | Add reference numerals (100–120 in FIG. 1; 200–224 in FIG. 2; add FIG. 3–5) and update descriptive text. | Substantive |
| **A8** | Section 7: Detailed Description (Module 3) | Describes feature configurations M1 (18), M2 (34), M3 (42), M4 (46). | Does not list the individual feature names, creating ambiguity about what constitutes each model. | `src/models/model_factory.py` lines 21–115 (`FEATURE_GROUPS`, `MODEL_CONFIGS`). | Provide exhaustive feature tables for M1, M2, M3, and M4 with names, physical meanings, and input cycle windows. | Substantive |
| **A9** | Section 7: Detailed Description (Module 4) | Describes peer matching ranking and distance. | Omits the explicit out-of-fold leave-one-out restriction for training reference cells ($N=40$ candidates). | `src/peers/peer_matcher.py` lines 205–213; `src/peers/peer_pipeline.py` lines 119–122. | Document that training cells exclude themselves from candidate matching, maintaining $N-1$ candidate pool. | Substantive |
| **A10** | Section 7: Detailed Description (Module 5) | Lists 8 peer-relative features. | Formulas, units, and missing value behavior are described only in narrative text without formal equations. | `src/peers/trajectory_comparison.py` lines 45–54, 128–145. | Include formal mathematical definitions, units (Ah, Ah/cycle, $\Omega$, unitless), and NaN handling rules. | Substantive |
| **A11** | Section 7: Detailed Description (Module 6) | States warning threshold: "peer-slope dispersion greater than $10^{-4}$ Ah per cycle". | Accurately matches code, but draft does not explain what actions occur when warnings trigger. | `src/peers/peer_reliability.py` lines 29–31, 168–180; `dashboard/app.js` warning alert banners. | Specify that warnings trigger diagnostic flags in metadata and advisory banners in the UI advising human engineering review or abstention. | Substantive |
| **A12** | Section 7: Detailed Description (Module 7) | States: "M4 additionally incorporates peer-reliability features... four PMRI component features". | In the implementation, M4 features are `pmri`, `peer_similarity_score`, `peer_agreement_score`, and `mean_peer_distance`. | `src/models/model_factory.py` lines 81–86. | Correct the statement to list the exact 4 features in `peer_reliability`. | Confirmed Error |
| **A13** | Section 7: Detailed Description (Module 8) | Mentions evaluation across separate cohorts, but cites no actual quantitative numbers, tables, or confidence intervals. | Lacks concrete experimental embodiment. Patent specifications require reproducible experimental support. | `results/phase5_model_metrics.csv`, `results/phase6_model_differences.csv`. | Insert comprehensive experimental tables showing verified AUROC, AUPRC, Brier score, and paired bootstrap differences. | Critical |
| **A14** | Section 7: Detailed Description (Module 9) | Describes interactive dashboard features. | Mentions planned or general capabilities without clarifying offline standard-library implementation. | `dashboard/server.py` (Python `http.server`, vanilla JS, zero external dependencies). | Describe actual implemented dashboard architecture (REST API, offline zero-npm, static Chart.js). | Substantive |
| **A15** | Section 10: Illustrative Example | Gives generic qualitative walkthrough for a hypothetical battery cell at Cutoff 70. | Does not provide concrete numerical values from a real test cell. | Real cell case studies in `results/phase4_peer_matching_report.md` (`b1c0` primary test, `b3c0` secondary test). | Ground the illustrative example using real benchmark cell `b1c0` (Cycle 70, protocol 3.6C(80%)-3.6C, peers `b1c1`, `b1c5`, PMRI 0.825). | Substantive |
| **A16** | Section 12: Advantages | Implies consistent predictive performance gains across all cohorts. | In-distribution results show statistical parity; gains occur specifically under domain/policy distribution shifts. | `results/phase6_hypothesis_evaluation_report.md` Section 2.1 & 2.2. | Accurately state advantages: domain-shift invariance, explicit peer unreliability detection, and interpretable trajectory visualization. | Substantive |
| **A17** | Claims 1–10 | Claims recite functional software modules without sufficient physical/technical interaction or concrete algorithmic steps. | Vulnerable to Section 3(k) rejection under Indian Patent Law as "computer programme per se". | Section 3(k) Patents Act 1970; *Ferid Allani* (2019) guidelines. | Restructure Claim 1 into a method/system operating on physical sensor signals, performing concrete transformation, and generating protective control/screening signals. | Critical |
| **A18** | Abstract | Summary paragraph describing system components. | Needs alignment with corrected claims, exact feature groupings, and technical contributions. | `Patent_DRAFT_1.docx` Abstract paragraph. | Update Abstract to reflect corrected feature counts, exact PMRI definition, and domain-shift robustness. | Substantive |

---

## 3. Deep Technical Audit by Subsystem

### 3.1 Peer Matching Subsystem (`src/peers/peer_matcher.py`)

#### Verified Technical Characteristics:
- **Matching Feature Set ($M=7$):**
  1. `init_policy_c1`: First fast-charging current rate ($C_1$, units: C-rate, weight: 0.20)
  2. `init_policy_soc`: State-of-charge switch threshold ($\text{SOC}_{\text{thresh}}$, units: %, weight: 0.15)
  3. `init_policy_c2`: Second fast-charging current rate ($C_2$, units: C-rate, weight: 0.15)
  4. `init_q_discharge`: Baseline discharge capacity at Cycle 2 ($Q_d(2)$, units: Ah, weight: 0.15)
  5. `init_internal_resistance`: Baseline internal resistance at Cycle 2 ($IR(2)$, units: $\Omega$, weight: 0.15)
  6. `init_temp_avg`: Baseline average surface temperature at Cycle 2 ($T_{\text{avg}}(2)$, units: $^\circ$C, weight: 0.10)
  7. `snap_internal_resistance`: High-frequency internal resistance at cutoff cycle $N_{\text{obs}}$ ($IR(N_{\text{obs}})$, units: $\Omega$, weight: 0.10)
  - **Weight Sum:** $0.20 + 0.15 + 0.15 + 0.15 + 0.15 + 0.10 + 0.10 = 1.00$.
- **Standardization:**
  $$z_{tj} = \frac{x_{tj} - \mu_j}{\sigma_j}$$
  where $\mu_j$ and $\sigma_j$ are computed **exclusively** on the 41 training reference pool cells at that specific observation cutoff. Test cells never influence $\mu_j$ or $\sigma_j$. Constant feature protection: if $\sigma_j < 10^{-8}$, $\sigma_j \leftarrow 1.0$.
- **Distance Metric:**
  Standardized Weighted Euclidean Distance:
  $$d(t, r) = \sqrt{\sum_{j=1}^7 w_j (z_{rj} - z_{tj})^2}$$
- **Peer Ranking and Selection:**
  - Candidates sorted in ascending order of $d(t, r)$.
  - Default target peer count: $k = 5$.
  - Minimum valid peer threshold: $k_{\min} = 3$.
  - **Self-Match Exclusion:** If candidate cell ID matches target cell ID ($r = t$), candidate is strictly omitted (`if ref_id == target_id: continue`). For training cells, candidate pool is strictly $N_{\text{train}} - 1 = 40$ cells.
  - **Reference Isolation:** Test batteries (primary and secondary) draw peers exclusively from the 41 training cells. No test cell may serve as a reference peer for any other cell.
- **Boundary & Fallback Behavior:**
  - If target cell is not benchmark-eligible or partition == 'excluded': returns `EXCLUDED_CELL`, $k=0$, $\bar{d} = \text{NaN}$.
  - If valid feature count $< M - 1$ (fewer than 6 valid features): returns `INSUFFICIENT_FEATURES`, $k=0$, $\bar{d} = \text{NaN}$.
  - If eligible candidates $< 3$: returns `INSUFFICIENT_PEERS`, $k = \text{count}$, $\bar{d} = \text{mean of available}$, status flagged as insufficient.

---

### 3.2 Peer-Relative Features Subsystem (`src/peers/trajectory_comparison.py`)

The draft describes 8 peer-relative features. The audit confirms that **all 8 features exist and are computed strictly over the observation interval $[2, N_{\text{obs}}]$**:

| Feature Name in Code | Mathematical Formula | Physical Input Variables | Engineering Units | Observation Requirement | Missing Value Handling |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `relative_capacity_deviation` | $\Delta Q_{\text{rel}} = Q_{\text{target}}(N_{\text{obs}}) - \text{median}_{p \in \text{peers}}(Q_p(N_{\text{obs}}))$ | Cutoff discharge capacity $Q_d$ of target and peers | Ah | $N_{\text{obs}} \ge 2$ | Returns `NaN` if peers empty or target non-finite |
| `relative_slope_deviation` | $\Delta (dQ/dn) = \left(\frac{dQ}{dn}\right)_{\text{target}} - \text{median}_{p \in \text{peers}}\left(\left(\frac{dQ}{dn}\right)_p\right)$ | Linear regression slope of $Q_d$ over cycles $[2, N_{\text{obs}}]$ | Ah/cycle | $N_{\text{obs}} \ge 3$ cycles | Returns `NaN` if $<3$ valid points |
| `relative_capacity_ratio` | $R_{Q} = \frac{Q_{\text{target}}(N_{\text{obs}})}{\text{median}_{p \in \text{peers}}(Q_p(N_{\text{obs}}))}$ | Cutoff discharge capacity $Q_d$ | Unitless (fraction) | $N_{\text{obs}} \ge 2$ | Returns `NaN` if peer median $\le 0$ |
| `relative_fade_deviation` | $\Delta Q_{\text{fade, rel}} = \Delta Q_{\text{fade, target}} - \text{median}_{p \in \text{peers}}(\Delta Q_{\text{fade}, p})$ | Total capacity fade: $Q_d(N_{\text{obs}}) - Q_d(2)$ | Ah | $N_{\text{obs}} \ge 2$ | Returns `NaN` if baseline or snapshot missing |
| `relative_ir_deviation` | $\Delta IR_{\text{rel}} = IR_{\text{target}}(N_{\text{obs}}) - \text{median}_{p \in \text{peers}}(IR_p(N_{\text{obs}}))$ | Cutoff internal resistance $IR(N_{\text{obs}})$ | $\Omega$ | $N_{\text{obs}} \ge 2$ | Returns `NaN` if peer IR unavailable |
| `peer_trajectory_dispersion` | $\sigma_{Q, \text{peers}} = \sqrt{\frac{1}{k} \sum_{p=1}^k (Q_p(N_{\text{obs}}) - \bar{Q}_{\text{peers}})^2}$ | Cutoff discharge capacity of all matched peers | Ah | $k \ge 1$ | Returns `NaN` if peer list empty |
| `peer_slope_dispersion` | $\sigma_{\text{slope, peers}} = \sqrt{\frac{1}{k} \sum_{p=1}^k \left(\left(\frac{dQ}{dn}\right)_p - \overline{\left(\frac{dQ}{dn}\right)}_{\text{peers}}\right)^2}$ | Degradation slopes $dQ/dn$ of all matched peers | Ah/cycle | $k \ge 1$, $\ge 3$ cycles | Returns `NaN` if slopes unavailable |
| `relative_z_score_capacity` | $z_{Q, \text{rel}} = \frac{Q_{\text{target}}(N_{\text{obs}}) - \bar{Q}_{\text{peers}}}{\sigma_{Q, \text{peers}} + 10^{-6}}$ | Target capacity, peer mean capacity, peer dispersion | Unitless (z-score) | $k \ge 1$ | Protected by $+10^{-6}$ denominator epsilon |

---

### 3.3 Peer-Match Reliability Index (PMRI) Subsystem (`src/peers/peer_reliability.py`)

#### Exact Mathematical Formulation:
The composite PMRI is defined as a convex linear combination of four bounded sub-components:
$$\text{PMRI} = 0.35 S_{\text{similarity}} + 0.30 S_{\text{agreement}} + 0.20 S_{\text{availability}} + 0.15 S_{\text{completeness}}$$

1. **Peer Similarity Score ($S_{\text{similarity}} \in [0.0, 1.0]$):**
   $$S_{\text{similarity}} = \exp\left(-\frac{\bar{d}}{d_0}\right)$$
   where $\bar{d} = \frac{1}{k} \sum_{p=1}^k d(t, p)$ is the mean standardized distance to the selected peers, and $d_0 = 2.0$ is the characteristic distance scale parameter.
   - Behavior: If $\bar{d} = 0$, $S_{\text{similarity}} = 1.0$. As distance increases to $2.0$, score decays to $e^{-1} \approx 0.368$. If $\bar{d}$ non-finite or negative, returns $0.0$.
2. **Trajectory Agreement Score ($S_{\text{agreement}} \in [0.0, 1.0]$):**
   $$S_{\text{agreement}} = \exp\left(-\frac{\sigma_{\text{slope, peers}}}{\sigma_0}\right)$$
   where $\sigma_{\text{slope, peers}}$ is the standard deviation of degradation slopes among the matched peers, and $\sigma_0 = 5 \times 10^{-5}$ Ah/cycle is the characteristic slope dispersion scale.
   - Behavior: If all peers degrade at identical slopes ($\sigma = 0$), $S_{\text{agreement}} = 1.0$. If dispersion reaches $5 \times 10^{-5}$ Ah/cycle, score decays to $e^{-1} \approx 0.368$.
3. **Peer Availability Score ($S_{\text{availability}} \in [0.0, 1.0]$):**
   $$S_{\text{availability}} = \begin{cases} 
   0.0, & \text{if } k < k_{\min} \ (k < 3) \\ 
   \min\left(1.0, \frac{k}{k_{\text{target}}}\right) = \frac{k}{5}, & \text{if } k \ge 3 
   \end{cases}$$
   - Behavior: Exactly 5 peers yields $1.0$; 4 peers yields $0.80$; 3 peers yields $0.60$; fewer than 3 peers yields $0.0$.
4. **Feature Completeness Score ($S_{\text{completeness}} \in [0.0, 1.0]$):**
   $$S_{\text{completeness}} = \frac{N_{\text{valid features}}}{M} = \frac{N_{\text{valid features}}}{7}$$
   - Bounded in $[0.0, 1.0]$.

#### Diagnostic Warning Thresholds:
The system evaluates four independent diagnostic warning conditions:
1. `LOW_PEER_COUNT`: $k < 5$ (fewer than target peers matched).
2. `HIGH_PEER_DISTANCE`: $\bar{d} > 2.0$ (mean standardized distance exceeds characteristic scale).
3. `HIGH_PEER_DISPERSION`: $\sigma_{\text{slope, peers}} > 1.0 \times 10^{-4}$ Ah/cycle ($2 \times \sigma_0$).
4. `LOW_RELIABILITY`: $\text{PMRI} < 0.50$.
- If no conditions are triggered, warning string is `"NONE"`.

#### Critical Conceptual Distinctions:
- **PMRI is NOT a failure probability:** It measures neighborhood density, similarity, and peer consensus in feature space.
- **PMRI is NOT a calibrated confidence of the logistic regression classifier:** It evaluates the empirical support for the peer-relative comparison.
- In M4, PMRI and its components are supplied as auxiliary input features, while in operational screening, PMRI functions as a **decision-support gate** alerting operators to abstain from automated classification when peer support is weak.

---

### 3.4 Target Variable & Prediction Horizon (`src/features/target_builder.py`)

#### Verified Ground Truth:
- **Nominal Cell Chemistry:** Commercial A123 Systems APR18650M1A cylindrical cells (LFP/graphite), nominal capacity = $1.1$ Ah.
- **EOL Threshold Definition:** First cycle where discharge capacity drops to $80\%$ of nominal capacity ($1.1 \times 0.80 = 0.88$ Ah).
- **Primary Prediction Target:**
  $$\text{target\_fail\_within\_500} = \begin{cases} 
  1, & \text{if } \text{EOL\_cycle} \le N_{\text{obs}} + 500 \\ 
  0, & \text{if } \text{EOL\_cycle} > N_{\text{obs}} + 500 
  \end{cases}$$
  - Cutoff 40: failure horizon is $\le 540$ cycles.
  - Cutoff 70: failure horizon is $\le 570$ cycles.
  - Cutoff 100: failure horizon is $\le 600$ cycles.
- **Alternative Prepared Targets (in Data Pipeline):**
  - `target_fail_within_300`: failure within $N_{\text{obs}} + 300$ cycles.
  - `target_fail_by_700`: absolute failure before cycle 700.
  - `target_rul`: continuous remaining useful life ($\text{EOL\_cycle} - N_{\text{obs}}$).
  - `target_eol_cycle`: absolute failure cycle.
- **Trained and Validated Scope:**
  Only `target_fail_within_500` was trained, calibrated, and validated across M1–M4. **Continuous RUL regression was NOT trained or validated in Phases 5–6.** The patent draft must explicitly state that continuous RUL is an alternative pipeline capability, not a validated classifier output.

---

### 3.5 Model Configurations & Algorithmic Pipelines (`src/models/model_factory.py`)

#### Confirmed Feature Counts and Hierarchy:
$$M_1 \subset M_2 \subset M_3 \subset M_4$$

1. **M1 (Condition Baseline — 18 features):**
   - Initial condition (`init_`, 11 features): `init_q_discharge`, `init_q_charge`, `init_coulombic_eff`, `init_internal_resistance`, `init_temp_max`, `init_temp_avg`, `init_temp_min`, `init_charge_time`, `init_policy_c1`, `init_policy_soc`, `init_policy_c2`.
   - Snapshot (`snap_`, 7 features): `snap_q_discharge`, `snap_q_charge`, `snap_coulombic_eff`, `snap_internal_resistance`, `snap_temp_max`, `snap_temp_avg`, `snap_charge_time`.
2. **M2 (Individual Trajectory — 34 features):**
   - M1 features (18) +
   - Trajectory (`traj_`, 11 features): `traj_q_fade`, `traj_q_retention`, `traj_q_slope`, `traj_q_curvature`, `traj_q_std`, `traj_q_min`, `traj_ir_change`, `traj_ir_slope`, `traj_temp_max_change`, `traj_temp_avg_mean`, `traj_charge_time_change`.
   - Differential curve (`curve_`, 5 features): `curve_delta_q_var_log10`, `curve_delta_q_min`, `curve_delta_q_mean`, `curve_delta_q_skew`, `curve_delta_q_kurtosis`.
3. **M3 (Peer-Relative Model — 42 features):**
   - M2 features (34) +
   - Peer-relative (`peer_traj`, 8 features): `relative_capacity_deviation`, `relative_slope_deviation`, `relative_capacity_ratio`, `relative_fade_deviation`, `relative_ir_deviation`, `peer_trajectory_dispersion`, `peer_slope_dispersion`, `relative_z_score_capacity`.
4. **M4 (Confidence-Aware Model — 46 features):**
   - M3 features (42) +
   - Peer-reliability (`peer_reliability`, 4 features): `pmri`, `peer_similarity_score`, `peer_agreement_score`, `mean_peer_distance`.

#### Estimator and Optimization Details:
- **Classifier:** `sklearn.linear_model.LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)`.
- **Preprocessing:** `sklearn.preprocessing.StandardScaler()`, fit strictly on training fold.
- **Decision Threshold Selection:** Youden's $J = \text{Sensitivity} + \text{Specificity} - 1 = \text{TPR} - \text{FPR}$, optimized on 5-fold out-of-fold training predictions, bounded in $[0.10, 0.90]$.
- **Probability Calibration:** Univariate Platt scaling (logistic regression on log-odds) fit strictly on 5-fold out-of-fold training predictions.
- **Secondary Benchmark:** Random Forest classifier implemented in `ModelFactory` (`n_estimators=100, max_depth=5, min_samples_split=4, class_weight='balanced'`).

---

## 4. Verification of Experimental Evidence (Phases 5 & 6)

### 4.1 Population Counts & Stratification Audit

| Partition | Total Raw Cells | Eligible Benchmark Cells | Excluded Cells | Cutoff 40 Positive / Neg (Prev) | Cutoff 70 Positive / Neg (Prev) | Cutoff 100 Positive / Neg (Prev) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 41 | 41 | 0 | 21 / 20 (51.2%) | 22 / 19 (53.7%) | 22 / 19 (53.7%) |
| **Primary Test** | 43 | 43 | 0 | 21 / 22 (48.8%) | 22 / 21 (51.2%) | 23 / 20 (53.5%) |
| **Secondary Test** | 40 | 40 | 0 | 0 / 40 (0.0%)* | 1 / 39 (2.5%) | 1 / 39 (2.5%) |
| **Excluded** | 16 | 0 | 16 | — | — | — |
| **Total** | **140** | **124** | **16** | **42 / 82** | **45 / 79** | **46 / 78** |

*\*Note on Secondary Test at Cutoff 40:* All 40 Batch 3 cells had cycle lives $> 1000$ cycles; the earliest failing cell failed at cycle 565. Since $565 > 40 + 500 = 540$, zero cells reached EOL within the 500-cycle horizon. Thus, prevalence is exactly 0%, and AUROC/AUPRC are mathematically undefined (`NaN`, flagged as `SINGLE_CLASS_EVALUATION`).

---

### 4.2 Verified Numerical Results: Primary Test Partition ($N=43$)

| Cutoff | Model | Features | AUROC | AUPRC | Brier Score | ECE | Threshold ($\tau$) | Sensitivity | Specificity | FPR | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **40** | **M1** | 18 | 0.9177 | 0.8461 | 0.0764 | 0.0886 | 0.36 | 0.9524 | 0.8636 | 0.1364 | 0.9091 |
| **40** | **M2** | 34 | 0.9091 | 0.8357 | 0.0871 | 0.1132 | 0.36 | 0.9048 | 0.9048 | 0.0952 | 0.9048 |
| **40** | **M3** | 42 | 0.8723 | 0.7834 | 0.1015 | 0.1471 | 0.34 | 0.9048 | 0.8636 | 0.1364 | 0.8837 |
| **40** | **M4** | 46 | 0.8680 | 0.7437 | 0.1033 | 0.1583 | 0.34 | 0.9048 | 0.8182 | 0.1818 | 0.8636 |
| **70** | **M1** | 18 | 0.9329 | 0.9106 | 0.0647 | 0.0784 | 0.34 | 1.0000 | 0.8571 | 0.1429 | 0.9362 |
| **70** | **M2** | 34 | 0.9156 | 0.8265 | 0.0620 | 0.0652 | 0.34 | 1.0000 | 0.8571 | 0.1429 | 0.9362 |
| **70** | **M3** | 42 | 0.9091 | 0.8082 | 0.0620 | 0.0645 | 0.34 | 1.0000 | 0.8571 | 0.1429 | 0.9362 |
| **70** | **M4** | 46 | 0.9091 | 0.8082 | 0.0628 | 0.0658 | 0.34 | 1.0000 | 0.8571 | 0.1429 | 0.9362 |
| **100** | **M1** | 18 | 0.9217 | 0.9114 | 0.0859 | 0.1032 | 0.34 | 0.9565 | 0.8500 | 0.1500 | 0.9167 |
| **100** | **M2** | 34 | 0.9152 | 0.8478 | 0.0807 | 0.0945 | 0.32 | 0.9565 | 0.8500 | 0.1500 | 0.9167 |
| **100** | **M3** | 42 | 0.9087 | 0.8378 | 0.0801 | 0.0938 | 0.30 | 0.9565 | 0.8500 | 0.1500 | 0.9167 |
| **100** | **M4** | 46 | 0.9065 | 0.8276 | 0.0815 | 0.0964 | 0.30 | 0.9565 | 0.8500 | 0.1500 | 0.9167 |

---

### 4.3 Verified Numerical Results: Secondary Test Partition ($N=40$, Out-of-Distribution)

| Cutoff | Model | Features | AUROC | AUPRC | Brier Score | Specificity | False Positive Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **40** | **M1** | 18 | `NaN`* | `NaN`* | 0.0310 | 0.9750 | 0.0250 |
| **40** | **M2** | 34 | `NaN`* | `NaN`* | 0.0249 | 0.9750 | 0.0250 |
| **40** | **M3** | 42 | `NaN`* | `NaN`* | 0.0256 | 0.9750 | 0.0250 |
| **40** | **M4** | 46 | `NaN`* | `NaN`* | 0.0252 | 0.9750 | 0.0250 |
| **70** | **M1** | 18 | **0.5128** | 0.0500 | 0.1428 | 0.8462 | 0.1538 |
| **70** | **M2** | 34 | 0.8462 | 0.1429 | 0.0503 | 0.9744 | 0.0256 |
| **70** | **M3** | 42 | **0.9744** | 0.5000 | 0.0650 | 0.9744 | 0.0256 |
| **70** | **M4** | 46 | **0.9744** | 0.5000 | 0.0928 | 0.9487 | 0.0513 |
| **100** | **M1** | 18 | **0.6667** | 0.0714 | 0.1401 | 0.8462 | 0.1538 |
| **100** | **M2** | 34 | 0.9487 | 0.3333 | 0.0412 | 0.9744 | 0.0256 |
| **100** | **M3** | 42 | **0.9744** | 0.5000 | 0.0425 | 0.9744 | 0.0256 |
| **100** | **M4** | 46 | **0.9744** | 0.5000 | 0.0812 | 0.9487 | 0.0513 |

*\*`NaN` indicates single-class ground truth (0 positive failure events).*

---

### 4.4 Paired Battery-Level Bootstrap Differences ($B=1000$ Resamples)

| Cutoff | Partition | Comparison | $\Delta\text{AUROC}$ Point | 95% Bootstrap CI | $\Delta\text{AUROC}$ Sig.? | $\Delta\text{Brier}$ Point | 95% Bootstrap CI | $\Delta\text{Brier}$ Sig.? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **40** | `primary_test` | `M1_to_M2` | -0.0087 | [-0.0411, +0.0152] | No (spans 0) | +0.0106 | [-0.0115, +0.0367] | No (spans 0) |
| **40** | `primary_test` | `M2_to_M3` | -0.0368 | [-0.1234, +0.0000] | No (spans 0) | +0.0144 | [-0.0003, +0.0427] | No (spans 0) |
| **40** | `primary_test` | `M3_to_M4` | -0.0043 | [-0.0195, +0.0000] | No (spans 0) | +0.0019 | [-0.0026, +0.0084] | No (spans 0) |
| **40** | `primary_test` | `M1_to_M4` | -0.0498 | [-0.1515, +0.0130] | No (spans 0) | +0.0269 | [-0.0062, +0.0797] | No (spans 0) |
| **70** | `primary_test` | `M1_to_M2` | -0.0173 | [-0.0649, +0.0108] | No (spans 0) | -0.0026 | [-0.0120, +0.0044] | No (spans 0) |
| **70** | `primary_test` | `M2_to_M3` | -0.0065 | [-0.0260, +0.0000] | No (spans 0) | -0.0001 | [-0.0027, +0.0026] | No (spans 0) |
| **70** | `primary_test` | `M3_to_M4` | 0.0000 | [0.0000, 0.0000] | No (spans 0) | +0.0009 | [-0.0010, +0.0039] | No (spans 0) |
| **70** | `primary_test` | `M1_to_M4` | -0.0238 | [-0.0715, +0.0043] | No (spans 0) | -0.0018 | [-0.0103, +0.0051] | No (spans 0) |
| **100** | `primary_test` | `M1_to_M2` | -0.0065 | [-0.0587, +0.0304] | No (spans 0) | -0.0052 | [-0.0144, +0.0033] | No (spans 0) |
| **100** | `primary_test` | `M2_to_M3` | -0.0065 | [-0.0261, +0.0000] | No (spans 0) | -0.0006 | [-0.0037, +0.0031] | No (spans 0) |
| **100** | `primary_test` | `M3_to_M4` | -0.0022 | [-0.0261, +0.0174] | No (spans 0) | +0.0015 | [-0.0007, +0.0048] | No (spans 0) |
| **100** | `primary_test` | `M1_to_M4` | -0.0152 | [-0.0696, +0.0196] | No (spans 0) | -0.0044 | [-0.0140, +0.0044] | No (spans 0) |
| **70** | `secondary_test` | `M1_to_M2` | +0.3333 | [+0.1795, +0.4872] | **Yes (Positive)** | -0.0925 | [-0.1250, -0.0620] | **Yes (Improved)** |
| **70** | `secondary_test` | `M2_to_M3` | +0.1282 | [+0.0513, +0.2308] | **Yes (Positive)** | +0.0147 | [-0.0050, +0.0350] | No |
| **70** | `secondary_test` | `M3_to_M4` | 0.0000 | [0.0000, 0.0000] | No | +0.0278 | [+0.0050, +0.0510] | Yes (Error Up) |
| **70** | `secondary_test` | `M1_to_M4` | +0.4615 | [+0.3077, +0.6154] | **Yes (Positive)** | -0.0500 | [-0.0820, -0.0190] | **Yes (Improved)** |

---

## 5. Audit Conclusions & Roadmap for Document Revision

1. **Retain Useful Structure:** Title, Keywords, Field, Background, Objectives, Summary, Drawings, Detailed Description, Illustrative Example, Advantages, Alternative Embodiments, Industrial Applicability, Claims, and Abstract are preserved.
2. **Correct Implementation Disclosures:**
   - Disclose exact 7 matching features and their physical weights ($0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10$).
   - Disclose exact non-linear PMRI formulas including scale parameters $d_0 = 2.0$ and $\sigma_0 = 5 \times 10^{-5}$ Ah/cycle.
   - Clarify M4's feature set (`pmri`, `peer_similarity_score`, `peer_agreement_score`, `mean_peer_distance`).
3. **Present Honest, Rigorous Scientific Findings:**
   - Clearly document that in-distribution, M1–M4 operate at statistical parity due to strong charging protocol signals in the Severson dataset.
   - Highlight the true technical contribution of peer-relative features: **robust generalization under policy and structural distribution shift** (+0.4616 AUROC gain over M1 on Batch 3).
   - Characterize PMRI as a **meta-reliability confidence gate and decision filter**, not an additive predictor.
4. **Prepare Revised Figures with Reference Numerals:**
   - Create high-resolution patent figures (FIG. 1 to FIG. 5) with standard reference numerals (100–500 series) and update the specification text to reference every numeral.
5. **Restructure Claims for Patentability:**
   - Redraft Claim 1 to recite a concrete physical-technical method/system operating on physical sensor signals, performing domain-shift normalization and PMRI confidence gating, avoiding Section 3(k) rejection.
