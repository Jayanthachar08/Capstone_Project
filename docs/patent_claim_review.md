# PATENT CLAIM AUDIT, SECTION 3(k) COMPLIANCE, AND PROPOSED CLAIM REVISION

**Document Audited:** `Patent_DRAFT_1.docx` (Claims 1–10)  
**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Repository:** `C:\engg notes\7th sem\Capstone_Project`  
**Date:** October 15, 2026  
**Auditor Roles:** Patent-Technical-Documentation Specialist, Senior ML Research Engineer, and Indian Patent Law Analyst  
**Applicable Legal Framework:** Indian Patents Act, 1970 (Section 3(k), Section 2(1)(j), Section 2(1)(ja), Section 10(4)), CRI Guidelines 2017, and Landmark Precedents (*Ferid Allani v. UOI*, *Microsoft v. Assistant Controller of Patents*).

---

## 1. Audit of Original Claims (Claims 1 to 10)

### Claim 1 (Original Independent System Claim):
> *"A computer-implemented system for early end-of-life risk prediction and peer-relative degradation analysis of lithium-ion battery cells, the system comprising: a battery data acquisition module configured to receive cycle-level battery measurements including discharge capacity, charge capacity, internal resistance, temperature measurements, charging time and charging-protocol parameters; a data preprocessing and feature engineering module configured to prepare battery records at a predefined observation cutoff and derive battery-condition and degradation-trajectory features; a peer-matching module configured to identify reference battery cells having similar initial-condition and operating characteristics to a target battery cell; a peer-relative degradation analysis module configured to determine deviations between the target battery cell and the identified reference battery cells; a Peer-Match Reliability Index (PMRI) module configured to assess the reliability of the peer comparison; and a machine-learning classification module configured to estimate whether the target battery cell will reach a predefined end-of-life threshold within a specified future cycle horizon based on the derived features."*

- **Specification Support:** Supported across Sections 1 to 7 of Detailed Description.
- **Implementation Status:** Verified across `src/peers/peer_pipeline.py` and `src/models/train_models.py`.
- **Defects & Legal Vulnerabilities:**
  1. *Section 3(k) Vulnerability:* Formulated as generic software modules ("data acquisition module", "feature engineering module", "classification module") performing mathematical operations. Under Indian Patent Office (IPO) examination practice, such claims are routinely objected to as "a computer programme per se or algorithms" lacking a tangible physical interaction or technical effect.
  2. *Broadness / Lack of Technical Interplay:* Merely lists modules in series without detailing the specific synergistic technical interaction between the peer-matching distance, the relative degradation deviations, and the PMRI confidence gating.
  3. *Unclaimed Technical Effect:* Does not state the practical, industrial output (e.g., generating protective control signals, battery pack derating signals, or safety isolation alerts).

---

### Claim 2 (Original Dependent System Claim — Peer Matching):
> *"The system as claimed in claim 1, wherein the said peer-matching module is configured to standardize selected matching features using parameters derived from training data, calculate a weighted Euclidean distance between the target battery cell and eligible reference battery cells, rank the reference battery cells according to the calculated distance, and select up to five nearest peers subject to a minimum peer-availability requirement, while excluding the target battery cell from its own reference group."*

- **Specification Support:** Fully supported by Detailed Description Section 4.
- **Implementation Status:** Verified in `src/peers/peer_matcher.py` lines 42–245.
- **Defects & Recommendations:**
  - Omission of the specific 7 matching features ($C_1, \text{SOC}, C_2, Q_d(2), IR(2), T_{\text{avg}}(2), IR(N_{\text{obs}})$) and their physical weights ($0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10$).
  - Recommendation: Retain as a dependent claim; explicitly recite the 7 physical matching features and their weighting structure to strengthen non-obviousness over general $k$-NN systems.

---

### Claim 3 (Original Dependent System Claim — Peer-Relative Features):
> *"The system as claimed in claim 1, wherein the said peer-relative degradation analysis module is configured to derive peer-relative features comprising target-cell capacity deviation from peer capacity, degradation-slope deviation, relative capacity ratio, capacity-fade deviation, internal-resistance deviation, peer-capacity dispersion, peer-slope dispersion, and standardized capacity deviation."*

- **Specification Support:** Fully supported by Detailed Description Section 5.
- **Implementation Status:** Verified in `src/peers/trajectory_comparison.py` lines 45–54.
- **Defects & Recommendations:**
  - Excellent technical specificity. All 8 features exist and are computed strictly over the early observation window $[2, N_{\text{obs}}]$.
  - Recommendation: Retain as a dependent claim; specify that median peer baselines are used to prevent outlier distortion.

---

### Claim 4 (Original Dependent System Claim — PMRI & Warnings):
> *"The system as claimed in claim 1, wherein the said Peer-Match Reliability Index (PMRI) module is configured to calculate a composite reliability score from a plurality of component scores comprising peer similarity, peer trajectory agreement, peer availability and matching-feature completeness, and to generate a reliability warning when one or more predefined conditions relating to peer count, peer distance, peer-slope dispersion or the calculated reliability score are satisfied."*

- **Specification Support:** Fully supported by Detailed Description Section 6.
- **Implementation Status:** Verified in `src/peers/peer_reliability.py` lines 69–188.
- **Defects & Recommendations:**
  - Does not recite the non-linear exponential decay formulations or the characteristic distance ($d_0 = 2.0$) and dispersion ($\sigma_0 = 5 \times 10^{-5}$ Ah/cycle) scales.
  - Recommendation: Retain as a dependent claim and add the governing exponential decay relationships.

---

### Claim 5 (Original Dependent System Claim — Model Configurations M1–M4):
> *"The system as claimed in claim 1, wherein the said machine-learning classification module comprises a plurality of feature configurations, including a baseline configuration using battery initial-condition and observation-cutoff features, a trajectory configuration incorporating individual degradation features, a peer-relative configuration incorporating peer-relative degradation features, and a confidence-aware configuration incorporating peer-reliability features, thereby enabling comparative evaluation of the contribution of the respective feature groups to end-of-life risk prediction."*

- **Specification Support:** Supported by Detailed Description Section 3 & 7.
- **Implementation Status:** Verified in `src/models/model_factory.py` lines 90–115.
- **Defects & Recommendations:**
  - Confirmed that M1=18, M2=34, M3=42, and M4=46 features.
  - Recommendation: Retain as a dependent claim; clarify that M4 incorporates PMRI, similarity score, agreement score, and mean distance as auxiliary features to evaluate incremental predictive contribution.

---

### Claim 6 (Original Dependent System Claim — Logistic Regression & Calibration):
> *"The system as claimed in claim 1, wherein the said machine-learning classification module is configured to apply a standardized logistic regression classifier with regularization and class weighting to estimate the probability that the target battery cell will reach an end-of-life threshold within a predefined future cycle horizon, and to apply probability calibration using predictions generated from training data."*

- **Specification Support:** Supported by Detailed Description Section 7.
- **Implementation Status:** Verified in `src/models/train_models.py` lines 128–188 and `src/models/calibration.py`.
- **Defects & Recommendations:**
  - Accurately describes the implemented L2-regularized logistic regression and Platt scaling fitted strictly on out-of-fold training predictions.
  - Recommendation: Retain as a dependent claim.

---

### Claim 7 (Original Dependent System Claim — EOL Threshold & Horizon):
> *"The system as claimed in claim 1, wherein the said predefined end-of-life threshold corresponds to the first cycle at which discharge capacity reaches 80% of nominal capacity, and wherein the said classification module is configured to determine whether the target battery cell reaches the said threshold within 500 cycles following a selected observation cutoff."*

- **Specification Support:** Supported by Detailed Description Section 7.
- **Implementation Status:** Verified in `src/features/target_builder.py` lines 18–26.
- **Defects & Recommendations:**
  - Clear and fully verified.
  - Recommendation: Retain as a dependent claim; add that the observation cutoff is selected from 40, 70, or 100 cycles.

---

### Claim 8 (Original Dependent System Claim — Interactive Dashboard):
> *"The system as claimed in claim 1, further comprising an interactive dashboard configured to present the predicted end-of-life risk, the corresponding classification, peer-relative degradation trajectories, peer-comparison information, PMRI values, reliability warnings and model evaluation visualisations."*

- **Specification Support:** Supported by Detailed Description Section 9.
- **Implementation Status:** Verified in `dashboard/server.py`, `dashboard/app.js`.
- **Defects & Recommendations:**
  - Recommendation: Retain as a dependent claim; emphasize the dynamic generation of the point-by-point peer centroid overlay curve.

---

### Claim 9 (Original Independent Method Claim):
> *"A computer-implemented method for early end-of-life risk screening of lithium-ion battery cells, the method comprising: receiving cycle-level battery measurements for a target battery cell and eligible reference battery cells; selecting a predefined observation cutoff; preparing battery-condition and degradation-trajectory features from measurements available up to the said cutoff; standardizing selected matching features using training-derived parameters; calculating similarity distances between the target battery cell and eligible reference battery cells; selecting a peer group according to the calculated distances and peer-availability requirements; deriving peer-relative degradation features; calculating a Peer-Match Reliability Index from peer similarity, trajectory agreement, peer availability and feature completeness; providing the resulting feature representation to a trained machine-learning classifier; estimating the probability that the target battery cell will reach a predefined end-of-life threshold within a specified future cycle horizon; and presenting the resulting risk estimate and peer-comparison reliability information through an interactive dashboard."*

- **Specification Support:** Supported by Detailed Description Section 10 and FIG. 2.
- **Implementation Status:** Verified in `src/peers/peer_pipeline.py`.
- **Defects & Legal Vulnerabilities:**
  - Suffers from the same Section 3(k) challenges as Claim 1. Framed as a series of mental/data-processing steps.
  - Recommendation: Restructure into a technical method for **operating a battery monitoring and degradation screening system**, explicitly reciting the acquisition of electrical signals from physical sensors, domain-shift normalization, confidence gating, and generating physical protective or advisory control outputs.

---

### Claim 10 (Original Dependent Method Claim — Anti-Leakage Isolation):
> *"The method as claimed in claim 9, wherein the training, preprocessing and probability-calibration procedures are performed using designated training data, and wherein held-out test battery cells are excluded from fitting the preprocessing and calibration parameters, thereby supporting cell-level evaluation of the end-of-life risk prediction system."*

- **Specification Support:** Supported by Detailed Description Section 2 & 11.
- **Implementation Status:** Verified in `tests/test_no_leakage.py` and `src/models/leakage_audit.py`.
- **Defects & Recommendations:**
  - Excellent integrity provision.
  - Recommendation: Retain as a dependent claim emphasizing reference pool isolation and self-exclusion.

---

## 2. Indian Patent Law Considerations (Section 3(k) & CRI Guidelines)

### 2.1 Statutory Provisions:
Section 3(k) of the Patents Act, 1970 excludes from patentability:
> *"a mathematical or business method or a computer programme per se or algorithms"*

### 2.2 Judicial Precedents and CRI Guidelines:
1. **Guidelines for Examination of Computer Related Inventions (CRI), 2017:**
   - The IPO Guidelines confirm that the presence of software does not automatically disqualify an invention.
   - The examiners must focus on the **underlying substance of the claim** rather than the form. If the claim exhibits a **technical effect** or **technical contribution** to an external physical entity or system, it is patent-eligible.
2. ***Ferid Allani v. Union of India* (Delhi High Court, 2019):**
   - The High Court ruled that the legislative intent behind adding "per se" was to avoid a blanket bar on computer-implemented inventions.
   - Inventions involving computer programs are patentable if they demonstrate a **"technical effect"** or a **"technical contribution"** (e.g., improved efficiency, physical transformation, machine control, or solving a technical problem).
3. ***Microsoft Technology Licensing v. Assistant Controller of Patents* (Delhi High Court, 2023):**
   - Affirmed that technical contribution can be established if the software interacts with hardware elements to produce an improved, secure, or novel operational result.

### 2.3 Application to the Present Battery Screening Invention:
- **The Technical Problem:** Non-destructive, early-cycle prediction of electrochemical failure in lithium-ion batteries across heterogeneous operating conditions, where distribution shifts across charging policies cause conventional models to fail silently without reference reliability indicators.
- **The Technical Effect / Contribution:**
  1. *Physical Domain Normalization:* Transforming physical voltage, current, and temperature sensor signals into protocol-normalized relative trajectory deviations that maintain predictive stability (AUROC 0.9744) across severe charging policy shifts where baseline models collapse (AUROC 0.5128).
  2. *Safety Confidence Gating & Physical Intervention:* Generating an independent physical reliability index (PMRI) that governs whether an automated screening decision is executed, preventing premature cell disposal or catastrophic unmonitored thermal runaway by generating automated protective derating or human-inspection control signals.
- **Claim Drafting Strategy:**
  - Avoid purely abstract algorithmic language.
  - Recite the interface with **physical sensors** (voltage, current, temperature sensors measuring an electrochemical cell).
  - Explicitly link the PMRI output to a **physical control or screening action** (e.g., transmitting an advisory alert to an energy management controller, gating an automated battery sorting mechanism, or adjusting a fast-charging current profile).

---

## 3. Proposed Revised Claim Set (Claims 1 to 12)

### Claim 1 (Independent System Claim — Technical Effect & Physical Framing):
**1.** A reliability-aware early end-of-life (EOL) screening and degradation monitoring system for lithium-ion battery cells, the system comprising:
- a battery sensor interface coupled to at least one sensor configured to measure physical operating parameters of a target lithium-ion battery cell, the measured parameters including cell terminal voltage, charge current, discharge current, surface temperature, and cycle indices across an early-life observation window bounded by a predefined observation cutoff cycle $N_{\text{obs}}$;
- a memory storing a reference pool of cycling degradation profiles associated with a plurality of historical reference battery cells, wherein the reference pool is restricted exclusively to designated training cells;
- one or more hardware processors configured to execute:
  - an initial feature extraction module that derives an initial operating vector from physical measurements at an early cycle, the vector comprising fast-charging current rates, state-of-charge switch thresholds, baseline discharge capacity, and baseline internal resistance;
  - an observation-cutoff feature extraction module that extracts snapshot features at the cutoff cycle $N_{\text{obs}}$, individual degradation trajectory slope features, and differential discharge voltage curve variance statistics over cycles preceding $N_{\text{obs}}$;
  - a peer-matching module configured to standardize the initial operating vector using statistics derived exclusively from the training reference pool, compute a standardized weighted Euclidean distance between the target cell and eligible reference cells in the reference pool while excluding the target cell from its own candidate set, rank the candidate cells in order of increasing distance, and select a peer cohort of $k$ nearest reference cells satisfying a minimum availability threshold $k_{\min} \ge 3$;
  - a peer-relative degradation analysis module configured to calculate a set of peer-relative features representing deviations between the target cell's degradation trajectory and corresponding degradation trajectories of the selected peer cohort over an aligned observation interval $[2, N_{\text{obs}}]$;
  - a Peer-Match Reliability Index (PMRI) module configured to calculate a composite reliability score $\text{PMRI} \in [0.0, 1.0]$ quantifying reference peer match suitability as a weighted sum of an exponential distance-decay peer similarity score, an exponential slope-dispersion-decay peer trajectory agreement score, a peer availability score, and a feature completeness score; and
  - an end-of-life risk screening classifier configured to process an input feature representation comprising the initial features, the snapshot features, the individual degradation trajectory features, and the peer-relative features to generate a calibrated probability that the target battery cell will reach an end-of-life capacity threshold within a predetermined future cycle horizon of 500 cycles following the observation cutoff cycle $N_{\text{obs}}$; and
- a controller output interface configured to issue a battery classification signal when the composite reliability score PMRI is equal to or greater than a predefined reliability threshold of 0.50, and to issue a peer-match advisory warning signal advising engineering review and decision abstention when the composite reliability score PMRI is less than 0.50.

---

### Claim 2 (Dependent System Claim — Matching Feature Weights):
**2.** The system as claimed in claim 1, wherein the initial operating vector evaluated by the peer-matching module consists of seven matching features:
- (i) a first-stage fast-charging current rate ($C_1$), assigned a normalized distance weight of 0.20;
- (ii) a state-of-charge switch threshold percentage ($\text{SOC}$), assigned a normalized distance weight of 0.15;
- (iii) a second-stage fast-charging current rate ($C_2$), assigned a normalized distance weight of 0.15;
- (iv) baseline discharge capacity measured at Cycle 2 ($Q_d(2)$), assigned a normalized distance weight of 0.15;
- (v) baseline internal resistance measured at Cycle 2 ($IR(2)$), assigned a normalized distance weight of 0.15;
- (vi) baseline average surface temperature measured at Cycle 2 ($T_{\text{avg}}(2)$), assigned a normalized distance weight of 0.10; and
- (vii) snapshot internal resistance measured at the observation cutoff cycle $N_{\text{obs}}$ ($IR(N_{\text{obs}})$), assigned a normalized distance weight of 0.10;  
wherein the sum of the normalized distance weights is exactly 1.00, and wherein standardization parameters comprising feature means and standard deviations are fitted exclusively on the designated training reference pool cells at the respective observation cutoff cycle.

---

### Claim 3 (Dependent System Claim — Peer-Relative Trajectory Features):
**3.** The system as claimed in claim 1, wherein the peer-relative degradation analysis module derives eight peer-relative features comprising:
- (i) relative capacity deviation, computed as the difference between target cell cutoff discharge capacity and median cutoff discharge capacity of the selected peer cohort;
- (ii) relative slope deviation, computed as the difference between target cell linear capacity degradation slope and median linear capacity degradation slope of the selected peer cohort over cycles 2 to $N_{\text{obs}}$;
- (iii) relative capacity ratio, computed as the ratio of target cell cutoff discharge capacity to median cutoff discharge capacity of the selected peer cohort;
- (iv) relative fade deviation, computed as the difference between target cell total capacity fade and median total capacity fade of the selected peer cohort from Cycle 2 to $N_{\text{obs}}$;
- (v) relative internal resistance deviation, computed as the difference between target cell cutoff internal resistance and median cutoff internal resistance of the selected peer cohort;
- (vi) peer trajectory dispersion, computed as the standard deviation of cutoff discharge capacities across the selected peer cohort;
- (vii) peer slope dispersion, computed as the standard deviation of linear capacity degradation slopes across the selected peer cohort; and
- (viii) relative capacity z-score, computed as the difference between target cell cutoff capacity and peer mean capacity divided by the sum of peer trajectory dispersion and an epsilon regularizer of $10^{-6}$.

---

### Claim 4 (Dependent System Claim — Exact PMRI Mathematical Decay Formulation):
**4.** The system as claimed in claim 1, wherein the Peer-Match Reliability Index (PMRI) module computes the composite reliability score according to:
$$\text{PMRI} = 0.35 S_{\text{similarity}} + 0.30 S_{\text{agreement}} + 0.20 S_{\text{availability}} + 0.15 S_{\text{completeness}}$$
wherein:
- $S_{\text{similarity}} = \exp(-\bar{d} / d_0)$, where $\bar{d}$ is the mean standardized weighted Euclidean distance to the selected peers and $d_0 = 2.0$ is a characteristic distance scale parameter;
- $S_{\text{agreement}} = \exp(-\sigma_{\text{slope}} / \sigma_0)$, where $\sigma_{\text{slope}}$ is the standard deviation of degradation slopes among the selected peers and $\sigma_0 = 5 \times 10^{-5}$ Ah/cycle is a characteristic slope dispersion scale parameter;
- $S_{\text{availability}} = \min(1.0, k / 5)$ when $k \ge 3$, and $S_{\text{availability}} = 0.0$ when $k < 3$; and
- $S_{\text{completeness}}$ is the ratio of valid non-missing matching features to the total count of seven matching features;  
and wherein the controller output interface generates diagnostic warning flags upon occurrence of any condition selected from: matched peer count $k < 5$; mean peer distance $\bar{d} > 2.0$; peer slope dispersion $\sigma_{\text{slope}} > 1.0 \times 10^{-4}$ Ah/cycle; and $\text{PMRI} < 0.50$.

---

### Claim 5 (Dependent System Claim — Model Configurations M1–M4):
**5.** The system as claimed in claim 1, wherein the end-of-life risk screening classifier supports four progressive feature configurations comprising:
- a condition baseline configuration (M1) utilizing 18 features consisting of 11 initial condition features and 7 snapshot features;
- an individual trajectory configuration (M2) utilizing 34 features consisting of the 18 M1 features, 11 individual trajectory features, and 5 differential discharge voltage curve variance statistics;
- a peer-relative configuration (M3) utilizing 42 features consisting of the 34 M2 features and the 8 peer-relative features; and
- a confidence-aware configuration (M4) utilizing 46 features consisting of the 42 M3 features and 4 peer-reliability features comprising the composite PMRI score, the peer similarity score, the peer agreement score, and the mean peer distance;  
wherein the peer-relative configuration (M3) and confidence-aware configuration (M4) provide invariant predictive performance across charging protocol distribution shifts where the condition baseline configuration (M1) degrades.

---

### Claim 6 (Dependent System Claim — Standardized Regularized Classifier & Calibration):
**6.** The system as claimed in claim 1, wherein the end-of-life risk screening classifier comprises:
- a standard scaler fitted exclusively on training-partition feature vectors;
- an L2-regularized logistic regression estimator with balanced class weighting;
- an optimal classification decision threshold $\tau$ selected to maximize Youden's $J$ statistic strictly on 5-fold out-of-fold training cross-validation predictions; and
- a univariate Platt scaling probability calibrator fitted on logits of the out-of-fold training cross-validation predictions to output calibrated probabilities bounded in $[0.0, 1.0]$.

---

### Claim 7 (Dependent System Claim — Target Definition & Observation Cutoffs):
**7.** The system as claimed in claim 1, wherein the predefined observation cutoff cycle $N_{\text{obs}}$ is selected from 40 cycles, 70 cycles, or 100 cycles, and wherein the end-of-life capacity threshold is defined as discharge capacity degrading to 80% of nominal cell capacity, corresponding to 0.88 Ah for a 1.1 Ah nominal lithium-iron-phosphate battery cell.

---

### Claim 8 (Dependent System Claim — Interactive UI & Centroid Overlay):
**8.** The system as claimed in claim 1, further comprising a visualization module configured to dynamically compute a point-by-point mean peer centroid trajectory across aligned cycles $[2, N_{\text{obs}}]$ from the selected peer cohort, and to render an interactive graphical display simultaneously overlaying:
- (i) the observed discharge capacity trajectory of the target battery cell;
- (ii) individual discharge capacity trajectories of the $k$ matched reference peer cells;
- (iii) the computed mean peer centroid trajectory;
- (iv) a calibrated risk probability and categorical risk badge; and
- (v) a radial gauge displaying the composite PMRI score alongside decomposed progress bars for the four sub-component scores.

---

### Claim 9 (Independent Method Claim — Technical Process & Control Output):
**9.** A computer-implemented method for early end-of-life risk screening and operational monitoring of lithium-ion battery cells, the method comprising:
- (a) receiving, via a hardware interface, cycle-indexed electrochemical measurements of a target lithium-ion battery cell from physical voltage, current, and temperature sensors up to an observation cutoff cycle $N_{\text{obs}}$ selected from 40, 70, or 100 cycles;
- (b) extracting an initial operating vector representing charging protocol parameters and Cycle 2 baseline health, and extracting cutoff snapshot features, degradation trajectory slopes, and differential discharge voltage curve variance statistics over cycles up to $N_{\text{obs}}$;
- (c) standardizing the initial operating vector using scaling statistics fitted strictly on a sequestered training reference pool of historical battery cells;
- (d) executing a weighted Euclidean distance calculation between the target cell and reference cells in the reference pool, excluding the target cell from its own candidate pool, and selecting $k$ nearest peer reference cells satisfying a minimum availability count $k_{\min} \ge 3$;
- (e) computing eight peer-relative degradation features comparing the target cell against median and dispersion statistics of the selected peer reference cells across an aligned early observation interval $[2, N_{\text{obs}}]$;
- (f) evaluating a Peer-Match Reliability Index (PMRI) combining exponential decay peer similarity, exponential decay peer slope agreement, peer availability, and feature completeness into a composite metric bounded in $[0.0, 1.0]$;
- (g) supplying the extracted features to a trained and calibrated classification model to estimate a calibrated probability of the target cell reaching an 80% nominal capacity end-of-life threshold within 500 cycles following the observation cutoff cycle $N_{\text{obs}}$; and
- (h) generating a physical screening signal, wherein:
  - when the PMRI score is equal to or greater than 0.50, the signal indicates an automated screening risk category for cell sorting or operational scheduling; and
  - when the PMRI score is less than 0.50, the signal issues an automated advisory alert indicating unverified peer matching and prompting engineering review or decision abstention.

---

### Claim 10 (Dependent Method Claim — Anti-Leakage Isolation):
**10.** The method as claimed in claim 9, wherein training data partitions, primary test data partitions, and secondary test data partitions are maintained strictly disjoint at the battery cell level, and wherein held-out test battery cells are strictly excluded from fitting standardization scalers, selecting decision thresholds, fitting probability calibrators, and populating reference peer candidate pools.

---

### Claim 11 (Dependent Method Claim — Domain Shift Adaptation):
**11.** The method as claimed in claim 9, wherein the target lithium-ion battery cell operates under a fast-charging protocol or structural packaging configuration exhibiting distribution shift relative to the reference pool, and wherein the computation of peer-relative degradation features normalizes the distribution shift, maintaining discriminative screening performance above 0.90 AUROC under conditions where un-normalized baseline models suffer performance degradation.

---

### Claim 12 (Dependent Method Claim — Battery Control System Integration):
**12.** The method as claimed in claim 9, wherein the generated physical screening signal is transmitted to a battery management system (BMS) or test-bench automation controller to automatically execute an action selected from: derating a maximum allowable charging current rate; triggering cell balancing; isolating the target cell from a battery pack; and logging an early-life warranty risk flag.

---

## 4. Legal Assessment & Next Steps

1. **Section 3(k) Defense:** By anchoring Claim 1 and Claim 9 in physical electrochemical sensor inputs, concrete feature transformations over aligned cycles, exponential consensus decay kernels, and physical screening/derating control outputs, the claim set avoids being characterized as an abstract algorithm or "computer programme per se". It establishes a clear **technical effect** under the *Ferid Allani* doctrine.
2. **Support in Specification:** Every element recited in Claims 1–12 is fully disclosed and supported by the Detailed Description, formulas, and experimental data in the revised draft.
3. **Mandatory Disclaimer:** While the proposed claim structure addresses the principal technical and statutory vulnerabilities of Draft 1, **this review constitutes technical and patent-engineering analysis, not formal legal advice**. Review and formal clearance by a registered Indian Patent Agent or Patent Attorney is strongly recommended prior to filing with the Indian Patent Office (IPO).
