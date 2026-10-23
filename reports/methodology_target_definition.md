# Methodology & Target Definition Specification

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Document:** Formulation of Primary Prognostics Target, Failure Horizon, and Benchmark Cohort Eligibility  
**Execution Date:** 2026-10-22  

---

## 1. Primary Prognostic Target Formulation

In lithium-ion battery prognostics, continuous cycle-life regression frequently suffers from high epistemic variance in early operational cycles due to flat initial capacity plateaus. To provide actionable, reliable screening for battery pack assembly and early qualification testing, this project frames early degradation screening as a binary risk classification task at observation cutoffs $N_{\text{obs}} \in \{40, 70, 100\}$ cycles.

### Mathematical Formulation: `target_fail_within_500`

The primary prognostic target $y_i(N_{\text{obs}})$ for battery cell $i$ observed at cutoff $N_{\text{obs}}$ is formally defined as:

$$
y_i(N_{\text{obs}}) = \begin{cases}
1, & \text{if } \text{EOL}_i - N_{\text{obs}} \le 500 \text{ cycles} \\
0, & \text{if } \text{EOL}_i - N_{\text{obs}} > 500 \text{ cycles}
\end{cases}
$$

Where:
- $\text{EOL}_i$ is the exact cycle index at which cell $i$'s discharge capacity falls to 80% of its nominal capacity ($0.88$ Ah for a 1.1 Ah nominal A123 Systems APR18650M1A cell).
- $\text{RUL}_i(N_{\text{obs}}) = \text{EOL}_i - N_{\text{obs}}$ is the ground-truth remaining useful life.
- A positive label ($y = 1$) indicates an *elevated risk / rapid degradation* battery destined to fail within 500 cycles of the observation point.
- A negative label ($y = 0$) indicates a *nominal longevity* battery that will survive beyond 500 cycles post-cutoff.

---

## 2. Dataset Cohort & Exclusion Criteria

The benchmark dataset comprises three batches of commercial A123 Systems 18650 LFP/graphite cells cycled under fast-charging policies in the MATR/Severson et al. benchmark (Nature Energy, 2019).

### Exclusion Protocol (16 Excluded Cells)
Out of 140 raw tested cells, exactly 16 cells are excluded in strict accordance with the published benchmark criteria:
1. **Channel / Cycler Interruptions:** Cells experiencing hardware communication dropouts or non-physical diagnostic spikes (e.g., `b1c4`, `b1c11`, `b1c12`, `b1c13`, `b1c17`, `b1c25`, `b1c34`, `b1c37`, `b1c39`, `b1c41`).
2. **Premature Termination (< 100 cycles):** Cells that terminated prior to cycle 100 (`b1c2`, `b1c3`, `b1c6`, `b1c7`, `b1c8`, `b1c9`) cannot be evaluated at $N_{\text{obs}} = 100$.

### Verified Benchmark Cohort (124 Cells)
The benchmark-eligible population consists of 124 cells:
- **Batch 1 (2017-05-12):** 46 raw cells $\rightarrow$ 31 eligible cells.
- **Batch 2 (2017-06-30):** 48 raw cells $\rightarrow$ 48 eligible cells.
- **Batch 3 (2018-04-12):** 46 raw cells $\rightarrow$ 45 eligible cells.

---

## 3. Class Balance and Target Distribution

Across the three evaluation cutoffs, the target distribution remains balanced across partitions:

| Partition | Total Cells | Cutoff 40 Positives | Cutoff 70 Positives | Cutoff 100 Positives | Baseline Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 41 | 22 ($53.7\%$) | 22 ($53.7\%$) | 21 ($51.2\%$) | $\sim 52.8\%$ |
| **Primary Test** | 43 | 21 ($48.8\%$) | 22 ($51.2\%$) | 23 ($53.5\%$) | $\sim 51.2\%$ |
| **Secondary Test** | 40 | 11 ($27.5\%$) | 11 ($27.5\%$) | 11 ($27.5\%$) | $\sim 27.5\%$ |
| **Total Benchmark** | 124 | 54 ($43.5\%$) | 55 ($44.4\%$) | 55 ($44.4\%$) | $\sim 44.1\%$ |

*Key Takeaway:* Train and Primary Test partitions possess balanced $\sim 50\%$ class distributions, ensuring that standard accuracy and balanced Youden's J decision thresholds are well-calibrated without artificial distortion from extreme class imbalance.
