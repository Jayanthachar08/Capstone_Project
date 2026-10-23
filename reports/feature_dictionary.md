# Comprehensive Feature Dictionary & Specification

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Document:** Complete Multi-Group Feature Taxonomy, Mathematical Formulations, Missing-Value Behavior, and Inference Availability  
**Scope:** All 46 Features for Models M1, M2, M3, M4  

---

## 1. Global Conventions & Physical Data Principles

1. **Cycle 1 Formation Artifact:** In the MATR/Severson dataset, cycle 1 is an unmonitored diagnostic/formation cycle with unrecorded capacity ($0.0$ Ah). The physical operational baseline is strictly set to **Cycle 2** ($Q_2, IR_2, T_2$).
2. **Strict Cutoff Isolation:** For any observation cutoff $N_{\text{obs}} \in \{40, 70, 100\}$, all calculations strictly utilize cycles $n \le N_{\text{obs}}$. Modifying or deleting cycle records with $n > N_{\text{obs}}$ has zero effect on any extracted feature.
3. **No Target Leakage:** No cycle life, end-of-life (EOL), remaining useful life (RUL), or failure status labels are accessible during feature calculation or peer matching.
4. **Missing-Value Policy:** Invalid mathematical operations (e.g., division by zero or $\log_{10}$ of non-positive variance) yield `np.nan` rather than arbitrary magic numbers (such as `-10.0`). Missing features are imputed strictly within training cross-validation folds using median imputation fit on training data only.

---

## 2. Model Feature Group Architecture

| Model Key | Model Name | Feature Groups Included | Total Features | Target Question |
| :---: | :--- | :--- | :---: | :--- |
| **M1** | Initial & Snapshot Baseline | `init` (11) + `snap` (7) | **18** | Can baseline condition and current snapshot alone predict EOL risk? |
| **M2** | Individual Trajectory | M1 (18) + `traj` (11) + `curve` (5) | **34** | Does individual degradation trajectory and $\Delta Q(V)$ curve improve prediction? |
| **M3** | Peer-Relative Trajectory | M2 (34) + `peer_traj` (8) | **42** | Do relative deviations from matched operational peers provide incremental signal? |
| **M4** | Confidence-Aware Peer-Relative | M3 (42) + `peer_reliability` (4) | **46** | Does incorporating peer-match reliability (PMRI) improve screening quality? |

---

## 3. Exhaustive Feature Specifications (46 Features)

### Group 1: Initial-Condition Features (`init_`, 11 Features)
*Measured at Cycle 2 (forming the battery's initial physical state).*

| Feature Name | Formulation / Source | Units | Window | Missing Value Policy | Inference Availability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `init_q_discharge` | $Q_2$ (Discharge capacity at cycle 2) | Ah | Cycle 2 | Fail/Exclude if missing | Measured at start of life |
| `init_q_charge` | $Q_{\text{charge}, 2}$ | Ah | Cycle 2 | Fail/Exclude if missing | Measured at start of life |
| `init_coulombic_eff` | $Q_2 / Q_{\text{charge}, 2}$ | Fraction | Cycle 2 | Impute median if $Q_{\text{chg}} \le 0$ | Measured at start of life |
| `init_internal_resistance`| $IR_2$ | $\Omega$ | Cycle 2 | Median train impute | Measured at start of life |
| `init_temp_max` | $\max(T_2(t))$ | °C | Cycle 2 | Median train impute | Measured at start of life |
| `init_temp_avg` | $\text{mean}(T_2(t))$ | °C | Cycle 2 | Median train impute | Measured at start of life |
| `init_temp_min` | $\min(T_2(t))$ | °C | Cycle 2 | Median train impute | Measured at start of life |
| `init_charge_time` | $t_{\text{charge}, 2}$ | min | Cycle 2 | Median train impute | Measured at start of life |
| `init_policy_c1` | First-stage fast charge rate | C-rate | Static | Fixed protocol specification | Known before cycling |
| `init_policy_soc` | Step-down SOC switch point | % | Static | Fixed protocol specification | Known before cycling |
| `init_policy_c2` | Second-stage charge rate | C-rate | Static | Fixed protocol specification | Known before cycling |

---

### Group 2: Instantaneous Snapshot Features (`snap_`, 7 Features)
*Instantaneous operating values observed at cycle $N_{\text{obs}}$.*

| Feature Name | Formulation / Source | Units | Window | Missing Value Policy | Inference Availability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `snap_q_discharge` | $Q_{N_{\text{obs}}}$ | Ah | Cycle $N_{\text{obs}}$ | Latest cycle $\le N_{\text{obs}}$ | Available at cutoff |
| `snap_q_charge` | $Q_{\text{charge}, N_{\text{obs}}}$ | Ah | Cycle $N_{\text{obs}}$ | Latest cycle $\le N_{\text{obs}}$ | Available at cutoff |
| `snap_coulombic_eff` | $Q_{N_{\text{obs}}} / Q_{\text{charge}, N_{\text{obs}}}$ | Fraction | Cycle $N_{\text{obs}}$ | Impute median if zero | Available at cutoff |
| `snap_internal_resistance`| $IR_{N_{\text{obs}}}$ | $\Omega$ | Cycle $N_{\text{obs}}$ | Median train impute | Available at cutoff |
| `snap_temp_max` | $\max(T_{N_{\text{obs}}}(t))$ | °C | Cycle $N_{\text{obs}}$ | Median train impute | Available at cutoff |
| `snap_temp_avg` | $\text{mean}(T_{N_{\text{obs}}}(t))$ | °C | Cycle $N_{\text{obs}}$ | Median train impute | Available at cutoff |
| `snap_charge_time` | $t_{\text{charge}, N_{\text{obs}}}$ | min | Cycle $N_{\text{obs}}$ | Median train impute | Available at cutoff |

---

### Group 3: Individual Degradation Trajectory Features (`traj_`, 11 Features)
*Dynamic degradation trajectory evaluated over cycles $n \in [2, N_{\text{obs}}]$.*

| Feature Name | Formulation / Source | Units | Window | Missing Value Policy | Inference Availability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `traj_q_fade` | $Q_2 - Q_{N_{\text{obs}}}$ | Ah | $[2, N_{\text{obs}}]$ | Zero if $N < 2$ | Available at cutoff |
| `traj_q_retention` | $Q_{N_{\text{obs}}} / Q_2$ | Fraction | $[2, N_{\text{obs}}]$ | 1.0 if $N < 2$ | Available at cutoff |
| `traj_q_slope` | Slope of linear fit to $Q(n)$ | Ah/cycle | $[2, N_{\text{obs}}]$ | Zero if $N < 3$ | Available at cutoff |
| `traj_q_curvature` | 2nd-order coefficient $a$ in $Q(n) = an^2+bn+c$ | Ah/cycle$^2$ | $[2, N_{\text{obs}}]$ | Zero if $N < 4$ | Available at cutoff |
| `traj_q_std` | $\text{std}(Q(n))$ | Ah | $[2, N_{\text{obs}}]$ | Zero if $N < 2$ | Available at cutoff |
| `traj_q_min` | $\min_{n \in [2, N_{\text{obs}}]} Q(n)$ | Ah | $[2, N_{\text{obs}}]$ | $Q_{N_{\text{obs}}}$ fallback | Available at cutoff |
| `traj_ir_change` | $IR_{N_{\text{obs}}} - IR_2$ | $\Omega$ | $[2, N_{\text{obs}}]$ | Zero fallback | Available at cutoff |
| `traj_ir_slope` | Slope of linear fit to $IR(n)$ | $\Omega$/cycle | $[2, N_{\text{obs}}]$ | Zero if $N < 3$ | Available at cutoff |
| `traj_temp_max_change` | $\max(T_{N_{\text{obs}}}) - \max(T_2)$ | °C | $[2, N_{\text{obs}}]$ | Zero fallback | Available at cutoff |
| `traj_temp_avg_mean` | $\text{mean}_{n}(\text{mean}(T_n))$ | °C | $[2, N_{\text{obs}}]$ | Snapshot fallback | Available at cutoff |
| `traj_charge_time_change`| $t_{\text{chg}, N_{\text{obs}}} - t_{\text{chg}, 2}$ | min | $[2, N_{\text{obs}}]$ | Zero fallback | Available at cutoff |

---

### Group 4: Differential Voltage Curve Statistics (`curve_`, 5 Features)
*Computed from high-resolution interpolated discharge curves $\Delta Q(V) = Q_{N_{\text{obs}}}(V) - Q_{10}(V)$ across 1,000 voltage points between 2.0 V and 3.5 V.*

| Feature Name | Formulation / Source | Units | Window | Missing Value Policy | Inference Availability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `curve_delta_q_var_log10` | $\log_{10}(\text{var}(\Delta Q(V)))$ | $\log_{10}(\text{Ah}^2)$ | $[10, N_{\text{obs}}]$ | `np.nan` if var $\le 0$; train median | Available at cutoff |
| `curve_delta_q_min` | $\min_V \Delta Q(V)$ | Ah | $[10, N_{\text{obs}}]$ | `np.nan` fallback | Available at cutoff |
| `curve_delta_q_mean` | $\text{mean}_V \Delta Q(V)$ | Ah | $[10, N_{\text{obs}}]$ | `np.nan` fallback | Available at cutoff |
| `curve_delta_q_skew` | $\text{skew}(\Delta Q(V))$ | Dimensionless | $[10, N_{\text{obs}}]$ | `np.nan` fallback | Available at cutoff |
| `curve_delta_q_kurtosis` | $\text{kurtosis}(\Delta Q(V))$ | Dimensionless | $[10, N_{\text{obs}}]$ | `np.nan` fallback | Available at cutoff |

---

### Group 5: Peer-Relative Trajectory Features (`peer_traj`, 8 Features)
*Quantifies trajectory deviations relative to $k=5$ reference peers selected strictly from the training cohort based on initial covariates.*

| Feature Name | Formulation / Source | Units | Window | Missing Value Policy | Inference Availability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `relative_capacity_deviation` | $Q_{\text{target}} - \text{median}(Q_{\text{peers}})$ | Ah | Cycle $N_{\text{obs}}$ | Zero if no peers | Available at cutoff |
| `relative_slope_deviation` | $\text{slope}_{\text{target}} - \text{median}(\text{slope}_{\text{peers}})$ | Ah/cycle | $[2, N_{\text{obs}}]$ | Zero if no peers | Available at cutoff |
| `relative_capacity_ratio` | $Q_{\text{target}} / \text{median}(Q_{\text{peers}})$ | Ratio | Cycle $N_{\text{obs}}$ | 1.0 if no peers | Available at cutoff |
| `relative_fade_deviation` | $\text{fade}_{\text{target}} - \text{median}(\text{fade}_{\text{peers}})$ | Ah | $[2, N_{\text{obs}}]$ | Zero if no peers | Available at cutoff |
| `relative_ir_deviation` | $IR_{\text{target}} - \text{median}(IR_{\text{peers}})$ | $\Omega$ | Cycle $N_{\text{obs}}$ | Zero if no peers | Available at cutoff |
| `peer_trajectory_dispersion` | $\text{IQR}(Q_{\text{peers}})$ or $\text{std}(Q_{\text{peers}})$ | Ah | Cycle $N_{\text{obs}}$ | Zero if no peers | Available at cutoff |
| `peer_slope_dispersion` | $\text{IQR}(\text{slope}_{\text{peers}})$ | Ah/cycle | $[2, N_{\text{obs}}]$ | Zero if no peers | Available at cutoff |
| `relative_z_score_capacity` | $\frac{Q_{\text{target}} - \text{mean}(Q_{\text{peers}})}{\text{std}(Q_{\text{peers}}) + \epsilon}$ | Z-score | Cycle $N_{\text{obs}}$ | Zero if std is 0 | Available at cutoff |

---

### Group 6: Peer-Match Reliability Features (`peer_reliability`, 4 Features)
*Composite and decomposed indicators measuring the reliability and geometric closeness of the reference peer set.*

| Feature Name | Formulation / Source | Units | Range | Missing Value Policy | Inference Availability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pmri` | Composite PMRI: $0.35 S_{\text{peer}} + 0.30 A_{\text{peer}} + 0.20 C_{\text{cond}} + 0.15 K_{\text{avail}}$ | Index | $[0.0, 1.0]$ | 0.0 with warning if no peers | Available at cutoff |
| `peer_similarity_score` | $S_{\text{peer}} = \exp(-d_{\text{mean}} / \tau_s)$ | Score | $[0.0, 1.0]$ | 0.0 if no peers | Available at cutoff |
| `peer_agreement_score` | $A_{\text{peer}} = \exp(-\sigma_{\text{peer}} / \tau_a)$ | Score | $[0.0, 1.0]$ | 0.0 if no peers | Available at cutoff |
| `mean_peer_distance` | Mean standardized Euclidean distance $d_{\text{mean}}$ | Normalized dist | $[0, \infty)$ | Inf / high if no peers | Available at cutoff |
