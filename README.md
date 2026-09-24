# Single-Cell-Transcriptomic-Classification-of-Tumour-Hypoxia
(Refer to the 'branches' dropdown to view each corresponding Jupyter notebook).
Single-cell classification of tumour hypoxia across two breast cancer cell lines and two sequencing platforms. Unsupervised manifold analysis (PCA, K-Means, Ward, UMAP), a tuned classifier suite (LogReg, RF, kNN, SVM, ensemble), cross-domain generalisation benchmarking, and MSigDB Hallmark enrichment of learned features.

# Single-Cell Transcriptomic Classification of Tumour Hypoxia

**A cross-platform, cross-cell-line machine learning framework for inferring the hypoxic state of individual breast cancer cells from scRNA-seq expression profiles — combined with post-hoc pathway-level interpretation of the learned decision function.**

---

## Overview

Hypoxia is a defining micro-environmental feature of solid tumours and a principal driver of metabolic reprogramming, therapeutic resistance and metastatic progression. This repository implements an end-to-end computational pipeline that determines whether the hypoxic transcriptional programme is (i) *detectable without supervision*, (ii) *predictable with supervision*, (iii) *transferable across sequencing chemistries and cell lines*, and (iv) *biologically interpretable* rather than an artefact of batch or platform.

The analysis spans **four single-cell datasets** defined by the factorial combination of two breast cancer cell lines (**MCF7**, **HCC1806**) and two sequencing technologies (**SMART-seq**, **Drop-seq**), each profiled under normoxic and hypoxic culture conditions. Every dataset is subjected to an identical modelling protocol, producing a **5 algorithms × 4 training sets = 20-model matrix**, which is then evaluated exhaustively **out-of-domain** to quantify generalisation under both biological and technical distribution shift.

---

## Scientific Objectives

1. **Unsupervised structure discovery** — establish whether oxygenation state is the dominant axis of variance in the expression manifold, without access to labels.
2. **Supervised discrimination** — train and rigorously tune a heterogeneous classifier suite to predict Hypoxia vs. Normoxia at single-cell resolution.
3. **Domain-transfer evaluation** — quantify degradation in predictive performance when a model trained on one (cell line × platform) domain is deployed on the remaining three.
4. **Mechanistic interpretability** — attribute the classifier's decision function to specific genes and test those genes for over-representation in curated biological pathways.

---

## Datasets

All matrices are pre-filtered, library-size normalised, log-transformed, and reduced to the **3,000 most highly variable genes** (HVG selection), supplied in genes × cells orientation and transposed to cells × genes for `scikit-learn` compatibility. Class labels are derived from the sample identifier string (`Norm` → 0, otherwise → 1).

| Dataset | Cell line | Technology | Cells | Features | Class balance | Sparsity |
|---|---|---|---|---|---|---|
| MCF7 SMART-seq | MCF7 | SMART-seq | 250 | 3,000 | 124 H / 126 N (balanced) | low |
| HCC1806 SMART-seq | HCC1806 | SMART-seq | 182 | 3,000 | balanced | low |
| MCF7 Drop-seq | MCF7 | Drop-seq | 21,626 | 3,000 | mild imbalance | high |
| HCC1806 Drop-seq | HCC1806 | Drop-seq | 14,682 | 3,000 | 61% H / 39% N | 97.64% |

Held-out **anonymised test sets** (labels withheld) are scored by the selected model and released as prediction artefacts.

---

## Repository Structure

```
.
├── MCF7_Unsupervised_Analysis (2).ipynb        # MCF7 SMART-seq — PCA, K-Means, Ward, UMAP
├── MCF7_Dropseq_Unsupervised_Analysis.ipynb    # MCF7 Drop-seq  — same protocol, noisier regime
├── Supervised_MCF7_SmartSeq_Niki.ipynb         # MCF7 SMART-seq — classifier suite + transfer
├── MCF7_Drop_Supervised_Analysis.ipynb         # MCF7 Drop-seq  — scaled to 17.3k training cells
├── Supervised_HCC1806_SmartSeq_Bianca.ipynb    # HCC1806 SMART-seq — yields the global best model
├── Supervised_HCC1806_DROPseq_Giulia.ipynb     # HCC1806 Drop-seq — imbalance-aware, F1-optimised
├── Pathway_Analysis_Bianca.ipynb               # Feature attribution + ORA against MSigDB Hallmark
├── Pathway_Analysis_ORA_Plot.png               # Enrichment bubble plot (−log10 FDR × overlap)
├── predictions_MCF7SmarSeq.csv                 # 63 anonymised cells
├── predictions_HCC1806SmarSeq.csv              # 182 anonymised cells
└── MCF7_Drop_predictions_1.csv                 # 5,406 anonymised cells
```

---

## Methodology

### 1. Preprocessing and representation

Identifier sanitisation (quote stripping), matrix transposition, label extraction from sample metadata, and **stratified 80/20 train–test partitioning** with a fixed random seed. Feature spaces are aligned across domains via `reindex(columns=X_train.columns, fill_value=0)`, guaranteeing that a model trained in one domain consumes an identically ordered, identically dimensioned feature vector in every other domain — a prerequisite for valid cross-dataset inference.

### 2. Unsupervised analysis

* **Standardisation** — `StandardScaler` (zero mean, unit variance per gene) to prevent high-magnitude transcripts from dominating the covariance structure.
* **PCA** — variance-retention criterion (`n_components=0.95`), with reconstruction error reported via inverse transform; scree/cumulative-variance diagnostics and 2D/3D/5D pairwise component visualisation.
* **K-Means** — model-order selection by the **elbow criterion** (inertia, `KElbowVisualizer`) cross-checked against **mean silhouette coefficient** and per-observation **silhouette analysis**; cluster **cardinality** and **magnitude** (Σ point-to-centroid distance) diagnostics.
* **Agglomerative clustering** — Ward linkage benchmarked against single, average and complete linkage; hierarchical structure rendered as a truncated dendrogram over the leading principal subspace.
* **UMAP** — non-linear manifold embedding to recover local structure that linear projections discard, with K-Means applied in the embedded space.
* **External validation** — clusters scored against withheld ground truth using **Adjusted Rand Index** and permutation-invariant clustering accuracy.

### 3. Supervised learning

Five estimators spanning distinct inductive biases:

| Model | Rationale | Tuned hyperparameters |
|---|---|---|
| **Logistic Regression** | Interpretable linear decision boundary; coefficients double as an attribution signal | `penalty` (L1/L2), `C` ∈ [1e-3, 1e3], `solver` (liblinear/saga) |
| **Random Forest** | Non-linear, variance-reducing, robust to overfitting; Gini importances | `n_estimators`, `max_depth`, `min_samples_leaf` |
| **k-Nearest Neighbours** | Non-parametric local density estimator; stress-tests the curse of dimensionality | `n_neighbors` (1–29), `weights`, `metric` (Euclidean/Manhattan) |
| **Support Vector Machine** | Margin maximisation in high-dimensional space; kernelised non-linearity | `kernel` (rbf/poly/sigmoid/linear), `C`, `gamma` |
| **Soft-Voting Ensemble** | Variance reduction through **F1-weighted** probability aggregation | per-estimator weights = validation F1 |

* **Model selection** — exhaustive `GridSearchCV` over the full Cartesian hyperparameter space with **5-fold stratified cross-validation**, parallelised across cores. Grid (rather than randomised) search was chosen deliberately to guarantee optimum coverage within each grid.
* **Scoring objective** — `accuracy` on balanced domains; **`f1`** on imbalanced domains (HCC1806 Drop-seq, 61/39), so that minority-class recall is not traded away for majority-class agreement.
* **Selective scaling** — distance- and margin-based learners (kNN, SVM) are wrapped in a `Pipeline` with `StandardScaler` so that normalisation statistics are fitted **inside each CV fold**, eliminating train–test leakage; tree ensembles and regularised linear models are fitted on the native normalised scale.
* **Computational strategy** — on the Drop-seq domains (17.3k and 14.7k training cells × 3,000 genes), quadratic-complexity learners (SVM, kNN) are trained on **stratified subsamples (n = 2,000)**; the sparsity profile (97.64% zeros) justifies subsampling as a variance-preserving approximation.
* **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves and precision–recall curves on the held-out partition.

### 4. Cross-domain generalisation protocol

Each of the 20 trained models is evaluated, **without retraining or recalibration**, on all four datasets. This isolates two orthogonal shift components: *biological* shift (MCF7 ↔ HCC1806) and *technical* shift (SMART-seq ↔ Drop-seq). Performance is aggregated into a model-selection heatmap and averaged across domains to identify the estimator with the most favourable robustness–accuracy trade-off.

### 5. Interpretability and pathway inference

The global best model's coefficient vector is ranked by magnitude (|β|) to extract the **top 150 discriminative genes**. These are submitted to **Over-Representation Analysis** (`gseapy.enrichr`) against the **complete MSigDB Hallmark collection (v2026.1, human)** — deliberately *not* restricted to hypoxia-associated sets, so that enrichment emerges from the data rather than from a confirmatory prior. Significance is assessed via Benjamini–Hochberg adjusted p-values and visualised as a bubble plot encoding −log₁₀(FDR), combined enrichment score and fractional overlap.

---

## Key Results

### Unsupervised: oxygenation is the dominant axis — but only at sufficient sequencing depth

| Metric | MCF7 SMART-seq | MCF7 Drop-seq |
|---|---|---|
| PCs for 95% variance | 205 | 2,654 |
| PC1 variance explained | 7.8% | 0.7% |
| K-Means (K=2) ARI | **0.952** (98.8% acc) | 0.505 |
| Ward agglomerative ARI | **0.984** (99.6% acc) | 0.387 |
| UMAP + K-Means accuracy | 97.2% | markedly > PCA |

Full-depth SMART-seq data recovers the biological condition **almost perfectly without labels** (99.6% agreement, 3 misassigned cells in 250). Drop-seq retains the signal but in a non-linear, low-SNR regime where UMAP substantially outperforms linear projection — a direct demonstration that platform chemistry, not biology, governs separability. K=3 solutions resolve two hypoxic sub-populations, consistent with graded adaptation to oxygen deprivation.

### Supervised: linear models dominate under distribution shift

* **MCF7 SMART-seq (in-domain):** all five classifiers attain perfect separation (accuracy = 1.000, AUC = 1.000) — expected given HVG preselection on a small, clean, balanced cohort.
* **MCF7 Drop-seq (in-domain):** Ensemble 98.5% → Logistic Regression 98.2% (AUC 0.998) → SVM 97.4% → Random Forest 96.8% → kNN 91.4%.
* **HCC1806 Drop-seq (in-domain):** Ensemble F1 = 0.965 (AUC 0.992); Logistic Regression F1 = 0.959; all models AUC > 0.98.
* **Out-of-domain:** non-linear and distance-based learners collapse catastrophically under technical shift, several degenerating to **zero recall** on the minority class (predicting a single class throughout). Logistic Regression is the **only estimator that never collapses**, retaining non-trivial precision, recall and AUC in every target domain.

**Selected global model — Logistic Regression trained on HCC1806 SMART-seq**, averaged across all four domains:

| Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| 84.4% | 87.1% | 79.2% | 81.6% | 90.6% |

### Pathway enrichment: the classifier learned biology, not batch

ORA of the top 150 coefficients against MSigDB Hallmark returns **HALLMARK_HYPOXIA as the single most significant gene set** (19/200 overlap, FDR ≈ 1.0 × 10⁻⁶, combined score 99.9), accompanied by a coherent downstream programme:

* **HALLMARK_MTORC1_SIGNALING** (17/200, FDR ≈ 2.0 × 10⁻⁴)
* **HALLMARK_GLYCOLYSIS** (14/200, FDR ≈ 1.2 × 10⁻³) — the canonical Pasteur-effect metabolic switch
* **HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY**, **HALLMARK_TNFA_SIGNALING_VIA_NFKB** — oxidative and inflammatory stress response
* **HALLMARK_P53_PATHWAY**, **HALLMARK_G2M_CHECKPOINT** — hypoxia-induced cell-cycle arrest
* **HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION**, **HALLMARK_APICAL_JUNCTION** — invasive/metastatic programmes
* **HALLMARK_ESTROGEN_RESPONSE_EARLY** — consistent with MCF7 ER⁺ lineage identity and hypoxia–hormone crosstalk

Gene-level attribution independently recovers glycolytic markers (**GAPDH**, **PGK1**, **ENO1**) and mitochondrial transcripts (**MT-RNR1/2**), confirming that the decision function is anchored in *bona fide* hypoxia physiology rather than in technical covariates.

---

## Prediction Artefacts

Each anonymised test set is scored by the selected model and exported with a uniform schema:

| Column | Type | Description |
|---|---|---|
| `sample` | str/int | Cell or library identifier |
| `prediction` | {0, 1} | 0 = Normoxia, 1 = Hypoxia |
| `probability_hypoxia` | float ∈ [0,1] | Calibrated posterior `predict_proba(X)[:,1]` |

`MCF7_Drop_predictions_1.csv` assigns 2,190 hypoxic and 3,216 normoxic cells (≈40/60) across 5,406 cells — a class ratio consistent with the training prior, indicating no degenerate bias toward either condition.

---

## Reproducibility

```bash
git clone <repository-url>
cd <repository>
pip install -r requirements.txt
jupyter lab
```

**Core stack:** `scikit-learn`, `pandas`, `numpy`, `scipy`, `umap-learn`, `yellowbrick`, `gseapy`, `matplotlib`, `seaborn`, `plotly`, `joblib`.

All stochastic components are seeded (`random_state = 42` / `123`) for deterministic reproduction. Input paths at the head of each notebook are local and must be repointed to your copy of the expression matrices; the Hallmark `.gmt` file is obtained from the [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb) portal.

---

## Limitations and Methodological Caveats

Reported transparently rather than suppressed:

* **Ensemble weight leakage.** Soft-voting weights are derived from F1 scores computed on the same held-out partition used for final ensemble evaluation, introducing mild optimistic bias. A nested validation split would be the correct remedy; inter-model comparisons remain valid.
* **Subsampling of quadratic learners.** SVM and kNN on Drop-seq domains are trained on n = 2,000 stratified subsamples, reducing their robustness under distribution shift and partly explaining their out-of-domain collapse.
* **Search cost.** Exhaustive grid search over 14.7k cells required ~13 h per linear/kernel model. `RandomizedSearchCV` combined with sparsity-justified subsampling would recover comparable optima at a fraction of the compute.
* **Ceiling effects.** Perfect in-domain scores on SMART-seq reflect HVG preselection on small cohorts and should be interpreted as evidence of separability, not of generalisation — which is precisely why the cross-domain protocol is the primary evaluation.
* **Platform confounding.** SMART-seq/Drop-seq differences are entangled with depth, sparsity and cohort size; disentangling them would require matched-depth downsampling experiments.

---

## Conclusions

The hypoxic transcriptional state is **recoverable without supervision at full sequencing depth**, **predictable with near-ceiling accuracy in-domain**, and **transferable across cell lines and chemistries only by appropriately regularised linear models**. Model complexity is negatively correlated with out-of-domain robustness in this regime: the simplest estimator is the only one that survives domain shift. Critically, pathway-level interrogation confirms that the resulting decision function encodes the canonical HIF-driven glycolytic, mTORC1 and stress-response programme — establishing the classifier as a biologically grounded instrument rather than a black-box discriminator.

---

## Contibution

Developed as a research project in mathematical modelling for machine learning (Bocconi University, AI Lab).

## License

Released under the MIT License. The underlying expression data remain subject to the terms of their original providers.
