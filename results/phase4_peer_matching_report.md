# Phase 4 Peer Matching & Reliability Evaluation Report

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Degree:** B.Tech Computer Science & Engineering (Capstone Project)  
**Execution Date:** 2026-10-12  
**Pipeline Version:** v1.0-weighted-euclidean-standardized  

---

## 1. Executive Summary

Phase 4 has implemented a reproducible, scientifically defensible peer-matching and peer-match reliability subsystem. For each battery cell observed across early-life cutoffs ($N_{\text{obs}} \in \{40, 70, 100\}$), the subsystem identifies top-5 reference peers strictly from the 41 training partition cells, computes 8 interpretable peer-relative degradation features over the compatible observation window, and calculates a separate 4-component Peer-Match Reliability Index (PMRI).

- **Total Records Processed:** 420 rows (140 cells $\times$ 3 cutoffs).
- **Benchmark Eligible Rows:** 372 rows (124 cells $\times$ 3 cutoffs).
- **Benchmark Excluded Rows:** 48 rows (16 cells $\times$ 3 cutoffs).
- **Match Success Rate in Eligible Cohort:** 100% (372 / 372).
- **Training-Test Isolation:** Strictly enforced. Primary and secondary test cells drew peers exclusively from the 41 training cells. Training cells used leave-one-out reference bank (40 candidates).

---

## 2. Methodology & Feature Definitions

### 2.1 Matching Feature Rationale & Weights
To prevent abnormal degradation trajectories from being normalized away, matching features capture operational stress regime and baseline manufacturing health rather than isolated capacity loss slope:

| Feature | Physical Meaning | Weight | Source Window |
| :--- | :--- | :---: | :--- |
| `init_policy_c1` | First-stage fast charge current rate (C) | 0.20 | Protocol (Cycle 2) |
| `init_policy_soc` | State-of-charge switch threshold (%) | 0.15 | Protocol (Cycle 2) |
| `init_policy_c2` | Second-stage charge current rate (C) | 0.15 | Protocol (Cycle 2) |
| `init_q_discharge` | Baseline discharge capacity at Cycle 2 (Ah) | 0.15 | Cycle 2 |
| `init_internal_resistance` | Baseline internal resistance at Cycle 2 ($\Omega$) | 0.15 | Cycle 2 |
| `init_temp_avg` | Baseline mean surface temperature ($^\circ$C) | 0.10 | Cycle 2 |
| `snap_internal_resistance` | High-frequency internal resistance at cutoff ($\Omega$) | 0.10 | Cycle $N_{\text{obs}}$ |

Standardization parameters (mean $\mu_f$ and standard deviation $\sigma_f$) are fitted **exclusively** on the permitted training reference cells at each respective cutoff.

### 2.2 Peer-Relative Trajectory Features
Compared over the strictly compatible early-life observation window $[2, N_{\text{obs}}]$:
- `relative_capacity_deviation`: $Q_{\text{target}}(N_{\text{obs}}) - \text{median}_{p \in \text{peers}}(Q_p(N_{\text{obs}}))$ [Ah]
- `relative_slope_deviation`: $(dQ/dn)_{\text{target}} - \text{median}_{p \in \text{peers}}((dQ/dn)_p)$ [Ah/cycle]
- `relative_capacity_ratio`: $Q_{\text{target}}(N_{\text{obs}}) / \text{median}_{p \in \text{peers}}(Q_p(N_{\text{obs}}))$ [unitless]
- `relative_fade_deviation`: $\Delta Q_{\text{fade, target}} - \text{median}_{p \in \text{peers}}(\Delta Q_{\text{fade}, p})$ [Ah]
- `relative_ir_deviation`: $IR_{\text{target}}(N_{\text{obs}}) - \text{median}_{p \in \text{peers}}(IR_p(N_{\text{obs}}))$ [$\Omega$]
- `peer_trajectory_dispersion`: $\text{std}_{p \in \text{peers}}(Q_p(N_{\text{obs}}))$ [Ah]
- `peer_slope_dispersion`: $\text{std}_{p \in \text{peers}}((dQ/dn)_p)$ [Ah/cycle]
- `relative_z_score_capacity`: $(Q_{\text{target}} - \bar{Q}_{\text{peers}}) / (\text{std}(Q_{\text{peers}}) + 10^{-6})$

### 2.3 Peer-Match Reliability Index (PMRI)
PMRI is a composite index bounded strictly in $[0.0, 1.0]$ measuring peer comparison suitability, **not** failure probability:

$$\text{PMRI} = 0.35 S_{\text{sim}} + 0.30 S_{\text{agree}} + 0.20 S_{\text{avail}} + 0.15 S_{\text{comp}}$$

- **Similarity ($S_{\text{sim}}$):** $\exp(-\bar{d} / 2.0)$, where $\bar{d}$ is the mean distance to selected peers.
- **Agreement ($S_{\text{agree}}$):** $\exp(-\sigma_{\text{peer\_slope}} / 5\times 10^{-5})$, rewarding tight consensus among peer slopes.
- **Availability ($S_{\text{avail}}$):** $\min(1.0, k_{\text{matched}} / 5)$.
- **Completeness ($S_{\text{comp}}$):** Fraction of valid matching features available.

---

## 3. Quantitative Results by Observation Cutoff

| Cutoff | Cohort | Count | Mean Distance | PMRI Mean $\pm$ Std | PMRI Min | PMRI Max | Clean Matches (% None) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **40** | Eligible Benchmark | 124 | 0.867 | 0.759 $\pm$ 0.089 | 0.514 | 0.892 | 98.4% |
| **70** | Eligible Benchmark | 124 | 0.868 | 0.766 $\pm$ 0.092 | 0.509 | 0.903 | 97.6% |
| **100** | Eligible Benchmark | 124 | 0.871 | 0.770 $\pm$ 0.096 | 0.500 | 0.911 | 96.8% |

### Partition-Specific PMRI Breakdown (Cutoff 100)

| Partition | Cells | Mean Peer Distance | Mean PMRI | Physical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `train` | 41 | 0.720 | 0.796 | Leave-one-out reference pool within Batch 1 & 2 |
| `primary_test` | 43 | 0.672 | 0.805 | Close protocol matching within Batch 1 & 2 |
| `secondary_test` | 40 | 1.239 | 0.707 | Inter-batch matching into Batch 3 (distribution shift captured by PMRI) |

---

## 4. Case Studies: Strong vs. Divergent Peer Matches

### Case 1: High Reliability Match (`b1c0`, Primary Test)
- **Charging Protocol:** 3.6C(80%)-3.6C
- **Selected Peers:** `b1c1`, `b1c5`, `b1c17`, `b2c29`, `b1c3`
- **Mean Peer Distance:** 0.614 (closest peer `b1c1` has distance 0.324 under identical protocol)
- **PMRI:** 0.825 (High reliability, Warning: `NONE`)
- **Relative Capacity Deviation:** +0.007 Ah (target cell retains more capacity than peer median)

### Case 2: Inter-Batch Match with Lower PMRI (`b3c0`, Secondary Test)
- **Charging Protocol:** 5C(67%)-4C-newstructure (Batch 3)
- **Selected Peers:** `b1c5`, `b1c21`, `b1c30`, `b1c32`, `b1c17` (Batch 1 & 2 training pool)
- **Mean Peer Distance:** 1.040
- **PMRI:** 0.705 (Lower similarity due to inter-batch structural shift, correctly reflected in PMRI)

---

## 5. Anti-Leakage & Integrity Verification

The implementation strictly adheres to all capstone research constraints:
1. **Reference Isolation:** Only the 41 training cells (`benchmark_partition == 'train'`) were permitted in the reference bank.
2. **Self-Exclusion:** Training cells were matched using leave-one-out out-of-fold reference pools ($N=40$).
3. **Cutoff Strictness:** Scaler statistics and matching features use only observations up to $N_{\text{obs}}$. Post-cutoff cycles were ignored.
4. **Target Independence:** No cycle life, RUL, failure horizon, or status columns were accessed during peer matching or PMRI calculation.

---

## 6. Planned Phase 5 Experiments

1. **M1 (Current Condition Baseline):** Train on `init_` and `snap_` features only.
2. **M2 (Individual Trajectory Baseline):** Train on `init_`, `snap_`, `traj_`, and `curve_` features.
3. **M3 (Peer-Relative Model):** Train with individual features plus peer-relative features (`relative_capacity_deviation`, `relative_slope_deviation`, etc.).
4. **M4 (Full Confidence-Aware Model):** Train with M3 features plus PMRI and reliability interaction terms.
5. **Ablation Studies:** Test varying $k \in \{3, 5, 7\}$ and evaluating prediction performance as a function of PMRI tiers.
