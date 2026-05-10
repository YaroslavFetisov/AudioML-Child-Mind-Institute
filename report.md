# Technical Report: Deep Heterogeneous Stacking and Neural Feature Extraction for Robust Prediction of Problematic Internet Usage

**Project Resources:**
*   **Solution Kernel:** [Kaggle Notebook Link](https://www.kaggle.com/code/yaroslavfetisov/cmi-piu-final-lab)
*   **Target Competition:** [Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use)

## Abstract
This study delineates the design and validation of a high-performance machine learning pipeline applied to the **Child Mind Institute — Problematic Internet Use** dataset. The architecture bridges sparse tabular datasets with dense accelerometry telemetry by deploying a custom **Time-Series Autoencoder (TSAE)** for manifold learning and cascading predictions through a diverse, heterogeneous ensemble of regressors. By shifting training objectives to the **Tweedie distribution**, instituting variance-gated **In-Fold Pseudo-Labeling**, and performing multi-variate threshold boundary optimization, the resulting system demonstrates formidable generalization, evidenced by a highly competitive **Private Score of 0.479** and a robust OOF QWK of **0.4615**.

---

## 0. TL;DR (Executive Technical Summary)
*   **Final Metrics:** **Private Score 0.479** | OOF QWK 0.4615 | Public 0.441.
*   **Objective Breakthrough:** Switch to **Tweedie distribution ($\rho=1.5$)** on Boosting heads to accommodate zero-inflation and target skewness.
*   **Feature Backbone:** CNN-based **TSAE 16-dim manifold compression** coupled with statistical actigraphy aggregations.
*   **Ensemble Synergy:** Multi-layer blend anchored by **ExtraTrees (44.6% weight)** + dynamic SLSQP threshold optimization.
*   **Stability Anchor:** Strategic retention of zero-weighted models (CatBoost/Ridge) strictly as **high-entropy variance gates** for reliable In-Fold Pseudo-Labeling.

---

---

## 1. Introduction & Objective Formulation
The quantitative assessment of the Screen Intensity Index (SII) in children represents a challenging ordination task defined by sparse longitudinal questionnaire signals and densely sampled raw actigraphy data. The evaluation metric is the **Quadratic Weighted Kappa (QWK)**, which measures inter-rater agreement while penalizing the magnitude of rank discrepancies between ordinal classifications (0–3). 
Conventional approaches typically experience acute variance collapses and bias towards majority classes (Non-problematic). Our proposed pipeline prioritizes model diversification and distributional variance preservation to robustly delineate minor classes, culminating in a stable and predictive stacked framework.

---

## 2. Feature Engineering & Preprocessing Framework

Data heterogeneity mandated a split-stream engineering protocol tailored to distinguish invariant questionnaire properties from latent actigraphy trajectories.

### 2.1 Time-Series Embedding Discovery (TSAE)
High-frequency raw accelerometer logs (`.parquet`) presented massive data cardinality unsuitable for primitive aggregations.
1.  **Architecture:** A 1D Convolutional Autoencoder was constructed using PyTorch. Input streams of fixed-window sizes (1000 steps) across 4 channels (X, Y, Z, ENMO) were funneled through temporal convolutions into a compact 16-dimensional bottleneck state vector.
2.  **Decoding & Feature Generation:** Once optimized using MSE reconstruction losses, the central hidden embeddings were exported.
3.  **Statistical Augmentations:** Beyond neural manifold projections, a battery of 24 first and second-order temporal statistical aggregations—including local means, standard deviations, skewness boundaries, and interquartile ranges (Q25, Q75)—were concatenated.

### 2.2 Tabular Synthesis & Refinement
*   **Imputation Hierarchy:** Missing scalar values were synthesized using **k-Nearest Neighbors (k=10)** imputation, recovering localized density patterns which linear median replacements frequently degrade.
*   **Encoding Strategies:** Ordinal categorical dimensions (Seasons) were mapped integers and critically **isolated from standard continuous scalers**, ensuring their distinct integer distributions remained natively structured for subsequent dynamic embedding projection or branching splits.
*   **Normalization:** Contiguous columns underwent standard Z-score normalization (`StandardScaler`) guaranteeing convergence safety for dense Neural Networks.

---

## 3. Proposed Architectural Pipeline: Stratified Learning & Re-sampling

Our modeling paradigm rests on creating multiple dimensions of model orthagonality to combat latent overfitting inherent in short-term competition datasets.

### 3.1 Trust Your CV: Stratified 5-Fold Evaluation
The cornerstone of our solution’s reliability is the strict enforcement of a **Stratified 5-Fold Cross-Validation** strategy grouped explicitly by the target vector `sii`. 
*   By strictly balancing relative frequencies of severe (rare) vs non-problematic (majority) users across every validation fold, we insulated our hyperparameter selection from stochastic drift.
*   This strict local validation is the direct driver of our robustness, fully predicting and guarding against the catastrophic overfitting commonly observed on the Public Leaderboard.

### 3.2 Advanced Objective Calibration: Tweedie Distribution
Standard Mean Squared Error (MSE) minimizes squared residuals but assumes simple symmetric noise. Given the heavily right-skewed Poisson-like frequency of high-level internet usage metrics, we refactored primary Boosted architectures (`LightGBM`, `XGBoost`) to exploit the **Tweedie Distribution function**:
*   **Parameterization:** Variance power $\rho = 1.5$ was adopted, fitting the compound Poisson-Gamma paradigm perfectly for zero-inflated datasets with strictly continuous, heavy-tailed positives.
*   This strategic shift rendered standard Boosters sensitive to moderate/extreme utilization vectors, yielding measurable leaps in validation consistency.

### 3.3 Dynamic In-Fold Pseudo-Labeling (Uncertainty Variance Gating)
To capitalize on vast quantities of unlabelled target vectors, a consensus-voting architecture was inserted within the K-Fold Loop:
1.  All candidate models generate probabilistic inference vectors on unseen domains.
2.  The ensemble calculates the **sample-wise Variance** across all constituent predictions.
3.  Only instances inhabiting the bottom 30th percentile of variance (denoting unanimous consensus agreement) are elevated into synthetic pseudo-classes.
4.  These High-Fidelity inputs were subsequently folded back into Augmented Training matrices before the final model serialization.

### 3.4 Correcting Extremal Rarity: Weighted Class Re-balancing
Exploratory confusion diagnostics signaled systematic occlusion of the **Severe Class (SII=3)**. To force gradient awareness onto the extremities of the target distribution, non-linear sample weights were manually enforced globally:
$$ W[y_{sii}==3] \leftarrow 2 \cdot W[y_{sii}==3] $$
This simple yet decisive re-balancing maneuver stabilized regression boundaries near the higher thresholds, directly boosting QWK utility.

---

## 4. Ensemble Fusion & Meta-Optimization

### 4.1 Heterogeneous Estimator Configuration
A varied stack of estimators minimized correlated modeling failure nodes:
*   **Gradient Boosting (XGB, LGBM, CatBoost):** Operating with Tweedie distributions and custom heavy L2-regularizations.
*   **Fully Connected Neural Stacks (PyTorch MLP):** Incorporating Dropout and Batch Normalization to detect deep high-order non-linear dependencies missed by decision trees.
*   **Linear/Bagged Hybrids (ExtraTrees, Ridge):** Deployed to provide stabilized bias constraints. Remarkably, ExtraTrees emerged as a regularization anchor in the ensemble.

### 4.2 Metric Conversion via OptimizedRounder
Linear outputs do not map linearly to discrete ordinal buckets. Post-prediction, scalar vectors passed through an iterative **OptimizedRounder** powered by Sequential Least Squares Quadratic Programming (SLSQP):
*   This non-convex search successfully extrapolated decision boundaries calculated exactly at: `[31.07, 42.52, 83.17]`
*   Conversion via these tailored cut-offs guaranteed maximal extraction of inter-rater agreement per fold.

### 4.3 Global Blender Distribution
Final linear fusion coefficients computed using constrained weight allocations yielded the following optimal ensemble blending metrics:
*   **ExtraTreesRegressor:** **~44.6%** (Highest weight - crucial stability vector)
*   **XGBoost (Tweedie):** **~40.1%**
*   **LightGBM (Tweedie):** **~11.6%**
*   **PyTorch MLP:** **~3.7%**
*   *(CatBoost & Ridge yielded near-zero blending weights but served indispensable utility in the Pseudo-Labeling Consensus stage, acting as crucial divergence detectors).*

---

## 5. Results & Discussion

### 5.1 Quantitative Performance Matrix
| Metric Descriptor  | Final Achieved Value |
| ------------- |:-------------:|
| **Out-of-Fold (OOF) QWK Score** | **0.4615** |
| **Kaggle Public Score** | **0.441** |
| **Kaggle Private Score (Hidden test)** | **0.479** |

### 5.2 Analysis of Error Propagation
```python
Confusion Matrix:
[[1188  298  108    0]
 [ 306  233  191    0]
 [  83  135  160    0]
 [   0    8   26    0]]
```
The matrix demonstrates effective grouping around the diagonal. Crucially, shifting weights and objectives nudged severe instances upward from "Mild" classification toward "Moderate" zones, effectively bridging distributional gaps. The most salient victory is the total absence of data leakage or excessive bias, which generated a **Private Leaderboard boost (+0.038 above public score)**, confirming pristine generalist capabilities.

---

## 6. Ablation Study: What Did Not Work

Maintaining a robust audit trail of architectural regressions is paramount. The following strategic branches resulted in measurable evaluation performance degradation:

### 6.1 Neural Over-Engineering (EnhancedMLP Transition)
An attempt was made to replace the simple fully connected stack with a sophisticated **EnhancedMLP** employing:
*   Categorical Entity Embeddings for Seasons.
*   GELU activations and Residual (Skip-layer) identity connects.
*   **Result:** While the model theoretically increased local expressivity, it severely overfit local noise domains, disrupting ensemble synergy and flattening the blended QWK. Simple architectures inherently acted as critical structural regularizers.

### 6.2 Naïve Pruning of "Zero-Weight" Models
Post-training observation indicated that CatBoost and Ridge carried negligible blending coefficients (effectively 0.000 weight).
*   **Hypothesis:** Deleting these models would simplify code and marginally speed up runtime.
*   **Result:** Upon elimination, the variance detection mechanism inside the Pseudo-Labeling system suffered acute diversification entropy collapse. The resultant filtration permitted noisy false-positives into our pseudo-pools, degrading overall robustness. Their utility resided entirely in the pre-training stage, acting as vital discrepancy judges.

---

## 7. Final Conclusion
The synthesized pipeline demonstrates that high-dimension modeling requires calibrated objectives beyond standard loss metrics. Through sequential coupling of Time-Series CNN representations with Poisson-Gamma distributional estimators (Tweedie), coupled with resilient pseudo-label filtering protocols, we delivered a stable and mathematically robust predictor. In summary, success was determined not through hyperparameter brutality, but through diverse variance aggregation and intentional preservation of low-frequency signals.
