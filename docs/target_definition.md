# Target Definition Specification (Phase 3)

**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  
**Degree:** B.Tech Computer Science & Engineering (Capstone Project)  
**Document:** End-of-Life Prediction Target & Horizon Formulation  

---

## 1. End-of-Life (EOL) Physical Criterion

In the MATR/Severson dataset (*Nature Energy* 2019), commercial cylindrical A123 Systems APR18650M1A lithium iron phosphate / graphite cells (nominal capacity $C_{\text{nom}} = 1.1$ Ah) were cycled to failure under diverse fast-charging protocols.

- **Authoritative Failure Criterion:** End-of-Life (EOL) is defined as the cycle number at which the battery's discharge capacity $Q_d$ first degrades to **$80\%$ of nominal capacity**:
  $$\text{Capacity Threshold} = 0.80 \times 1.1\text{ Ah} = 0.88\text{ Ah}$$
- **Recorded Cycle Life Variable:** For all 124 eligible benchmark cells, testing was continued until this failure criterion was met, yielding an uncensored integer cycle count $y = \text{benchmark\_cycle\_life}$ ranging from **148 to 2,237 cycles** (mean: 801.6, median: 736.5 cycles).
- **Excluded/Censored Cells:** Cells whose tests were discontinued prematurely or whose cycler channels failed prior to reaching $80\%$ capacity (e.g. `b3c23`, `b3c32`) are strictly labeled as **`censored`** with `NaN` target values.

---

## 2. Prediction Targets Formulation

### 2.1 Primary Target: Horizon-Specific Relative EOL Risk ($H = 500$ Cycles)
- **Mathematical Definition:**
  $$Y_{500}(N_{\text{obs}}) = \begin{cases} 1 & \text{if } \text{EOL} \le N_{\text{obs}} + 500 \\ 0 & \text{if } \text{EOL} > N_{\text{obs}} + 500 \end{cases}$$
- **Rationale:** Evaluates whether a battery is at risk of failing within the next 500 operational cycles following the observation cutoff $N_{\text{obs}}$.
- **Empirical Class Balance across Cutoffs (Eligible Cohort $N=124$):**
  - **Cutoff 40:** Positive = 42 ($33.9\%$), Negative = 82 ($66.1\%$)
  - **Cutoff 70:** Positive = 45 ($36.3\%$), Negative = 79 ($63.7\%$)
  - **Cutoff 100:** Positive = 46 ($37.1\%$), Negative = 78 ($62.9\%$)
- **Suitability:** Highly suitable for probability calibration, AUROC/AUPRC evaluation, and early-warning screening without degenerate class imbalance.

---

### 2.2 Secondary Target 1: Short-Term Relative Risk ($H = 300$ Cycles)
- **Mathematical Definition:**
  $$Y_{300}(N_{\text{obs}}) = \begin{cases} 1 & \text{if } \text{EOL} \le N_{\text{obs}} + 300 \\ 0 & \text{if } \text{EOL} > N_{\text{obs}} + 300 \end{cases}$$
- **Empirical Class Balance across Cutoffs ($N=124$):**
  - **Cutoff 40:** Positive = 3 ($2.4\%$), Negative = 121 ($97.6\%$)
  - **Cutoff 70:** Positive = 3 ($2.4\%$), Negative = 121 ($97.6\%$)
  - **Cutoff 100:** Positive = 3 ($2.4\%$), Negative = 121 ($97.6\%$)
- **Scientific Limitation & Caution:** Because only 3 batteries in the entire dataset fail within 400 cycles (`b2c1`, `b2c2`, `b2c24`), this target exhibits **extreme class imbalance** (less than $3\%$ positive). In the training split of 41 cells, only 1 cell is positive. While maintained for exploratory short-horizon analysis, it is ill-suited as a primary evaluation target.

---

### 2.3 Secondary Target 2: Absolute Early Failure ($\text{EOL} \le 700$ Cycles)
- **Mathematical Definition:**
  $$Y_{\text{early}} = \begin{cases} 1 & \text{if } \text{EOL} \le 700 \\ 0 & \text{if } \text{EOL} > 700 \end{cases}$$
- **Rationale:** Identifies batteries that fail in the lower quartile/half of the population (near the cohort median of 736.5 cycles), regardless of cutoff.
- **Empirical Class Balance:** Positive = 54 ($43.5\%$), Negative = 70 ($56.5\%$) across all cutoffs.
- **Suitability:** Exceptionally well balanced for benchmarking early screening classifiers across cutoffs.

---

### 2.4 Continuous Regression Targets
1. **Total Cycle Life:** $y = \text{benchmark\_cycle\_life}$
2. **Remaining Useful Life (RUL) at Cutoff:**
   $$\text{RUL}(N_{\text{obs}}) = \text{benchmark\_cycle\_life} - N_{\text{obs}}$$
   - Minimum RUL at Cutoff 40: $148 - 40 = 108$ cycles
   - Minimum RUL at Cutoff 70: $148 - 70 = 78$ cycles
   - Minimum RUL at Cutoff 100: $148 - 100 = 48$ cycles
   - All 124 eligible cells remain in active operation at all three observation cutoffs (no cell failed prior to cycle 100).

---

## 3. Labeling Rules & Censoring Handling

For any observation row:
1. **Positive (`positive`, label = 1):** The cell is verified benchmark-eligible and its observed EOL event occurs within the horizon.
2. **Negative (`negative`, label = 0):** The cell is verified benchmark-eligible and its observed EOL cycle exceeds the horizon.
3. **Censored (`censored`, label = NaN):** The cell was excluded from the benchmark (e.g. test was terminated before failure or channel was noisy). These rows are strictly masked during supervised training and loss computation.

---

## 4. Leakage Prevention Protocol

- **Target Masking:** Target columns (`target_...` and `status_...`) are strictly sequestered from feature sets.
- **No Target Leakage in Features:** Feature calculation pipelines never receive cycle life, future cycle counts, or target outcomes. Automated tests verify zero overlap between feature names and target keywords.
