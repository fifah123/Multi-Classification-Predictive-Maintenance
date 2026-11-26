# Industrial Equipment Fault Detection & Multi-Class Classification (AI4I 2020)

An end-to-end machine learning project diagnosing multi-class equipment failure modes using real-world industrial milling sensor data from the **UCI AI4I 2020 Predictive Maintenance Dataset**.

---

## 1. Project Overview
Predictive maintenance requires identifying *which* specific failure mechanism is occurring—not just whether a machine will fail—so maintenance teams can dispatch the right intervention. 

This project covers:
1. **Exploratory Data Analysis (EDA) & Label Hygiene:** Resolving documented target anomalies and extreme class imbalance (`~96.6%` healthy vs. `~3.4%` failure states).
2. **Domain-Physics Feature Engineering:** Translating thermodynamic and mechanical relationships into predictive features (`Power_Watts`, `Temp_Diff`, `Overstrain_Index`).
3. **Model Benchmarking:** Comparing a linear baseline (**Logistic Regression**) against a non-linear gradient-boosted ensemble (**XGBoost**) using imbalance-aware metrics (**Macro F1** and **Matthews Correlation Coefficient**).

---

## 2. Dataset & Preprocessing
* **Source:** [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset (ID: 601)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
* **Raw Size:** 10,000 operational records across 6 raw sensor/metadata features (`Type`, `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear`).
* **Data Cleaning & Target Construction:**
  * Removed 9 rows flagged as `Machine failure = 1` with no corresponding failure mode.
  * Removed 18 rows flagged as `RNF = 1` while `Machine failure = 0`.
  * Resolved rare simultaneous failures using physical severity priority (`OSF > PWF > HDF > TWF > RNF`).

### Target Classes
| Class Code | Failure Mode | Description |
| :--- | :--- | :--- |
| **`No_Failure`** | Normal Operation | Healthy machine operating within normal tolerances (~96.6% of dataset). |
| **`HDF`** | Heat Dissipation Failure | Triggered when temperature delta is low (`< 8.6 K`) alongside low rotational speed (`< 1380 rpm`). |
| **`PWF`** | Power Failure | Triggered when mechanical power falls below `3,500 W` or exceeds `9,000 W`. |
| **`OSF`** | Overstrain Failure | Triggered when the product of `Tool wear` and `Torque` exceeds product-grade thresholds (`L`, `M`, `H`). |
| **`TWF`** | Tool Wear Failure | Stochastic failure occurring within the `200–240 min` tool wear window. |

---

## 3. Domain-Physics Feature Engineering
Raw sensor readings alone require deep decision boundaries to approximate physical laws. Six domain features were engineered to linearize and isolate failure regimes:

* **`Type_Encoded`:** Ordinal encoding of product quality grade (`L = 0`, `M = 1`, `H = 2`).
* **`Temp_Diff` & `Temp_Ratio`:** $\Delta T = T_{\text{process}} - T_{\text{air}}$ to capture heat dissipation efficiency (`HDF`).
* **`Power_Watts`:** Mechanical power $P = \tau \cdot \omega = \text{Torque} \times \left(\text{RPM} \times \frac{2\pi}{60}\right)$ to isolate power boundaries (`PWF`).
* **`Overstrain_Index`:** $\text{Tool wear} \times \text{Torque}$ to capture cumulative mechanical strain (`OSF`).
* **`Wear_Per_RPM`, `Torque_Per_RPM`, `Thermal_Stress`:** Secondary interaction ratios capturing combined thermal and mechanical load.

---

## 4. Model Benchmark & Experimental Results

Both models were evaluated on a stratified 80/20 train-test split (`n_test = 1,995`) using balanced class/sample weighting to address severe class imbalance.

### Overall Comparison
| Model | Macro F1 (Test) | MCC (Test) | Accuracy | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (`StandardScaler` + L2) | 0.5263 | 0.4139 | 86.17% | 0.4485 | **0.9122** |
| **XGBoost Classifier** (`multi:softprob`) | **0.7897** | **0.8261** | **98.80%** | **0.7697** | 0.8133 |

> **Cross-Validation Stability:** XGBoost achieved a **5-Fold Stratified CV Macro F1 of `0.8023 ± 0.0134`**, confirming consistent generalization across folds.

### Per-Class F1-Score Breakdown
| Class | Test Support | Logistic Regression F1 | XGBoost F1 | Key Diagnostic Insight |
| :--- | :---: | :---: | :---: | :--- |
| **`No_Failure`** | 1,929 | 0.9238 | **0.9938** | XGBoost reduces false alarms on healthy machines by ~94%. |
| **`PWF`** | 17 | 0.5075 | **0.9714** | `Power_Watts` enables near-perfect isolation of two-sided power limits. |
| **`OSF`** | 20 | 0.8000 | **0.9500** | `Overstrain_Index` provides strong separation in both linear and tree models. |
| **`HDF`** | 21 | 0.3279 | **0.9333** | Tree splits capture the joint `Temp_Diff` + `RPM` condition without false positives. |
| **`TWF`** | 8 | 0.0723 | **0.1000** | Primary bottleneck due to extreme sample scarcity (8 test cases) and stochastic wear overlap. |

---

## 5. Key Findings & Conclusions
1. **Why Non-Linear Models Win:** Industrial failure rules rely on two-sided bounds (`PWF`: low *or* high power) and conditional intersections (`HDF`: low temperature delta *and* low RPM). Linear hyperplanes in Logistic Regression cast too wide a net, generating ~273 false alarms on healthy machines, whereas XGBoost's axis-aligned splits isolate bounded operating windows cleanly (`MCC = 0.8261`).
2. **Impact of Physics Features:** Engineering `Overstrain_Index` and `Power_Watts` enabled even a simple linear classifier to achieve **100% recall** on `OSF` and `PWF`, and propelled XGBoost above `0.93 F1` on three of the four deterministic failure modes.
3. **The `TWF` Bottleneck:** Unlike `PWF`, `OSF`, and `HDF`, Tool Wear Failure (`TWF`) is generated stochastically in the `200–240 min` window and has minimal support (`8` test samples). Aggressive sample weighting causes over-prediction in high-wear normal zones.

---

## 6. Project Structure & Setup

### Installation
```bash
pip install ucimlrepo pandas numpy scikit-learn xgboost matplotlib seaborn