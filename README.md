# Parkinson's Disease Detection — Advanced Machine Learning Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f89a36?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189fdd?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-7c3aed?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-3.4%2B-00aaff?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## Overview

This project implements a research-grade, end-to-end machine learning pipeline for the early detection of Parkinson's Disease from sustained phonation voice biomarkers. The system is built on the UCI Parkinson's Voice Dataset (Little et al., 2008) and extends standard classification methodology with Bayesian hyperparameter optimisation via Optuna, SHAP-based model explainability, deep neural network architecture search, statistical significance testing across classifiers, model calibration analysis, and cross-dataset generalization evaluation against the UCI Parkinson's Telemonitoring Dataset.

All results are presented through an interactive ten-tab web dashboard deployed on Streamlit, enabling reproducible exploration of the entire ML pipeline from raw data through to live diagnosis simulation.

---

## Live Application

> Add your Streamlit Cloud deployment URL here after deploying.

Deployment instructions are provided in the [Deployment](#deployment) section below.

---

## Research Motivation

Parkinson's Disease affects an estimated ten million people globally. Clinical diagnosis in early stages is notoriously unreliable due to symptom overlap with other movement disorders. Sustained phonation voice recordings — captured non-invasively over a telephone line — have been shown to contain measurable biomarkers of PD-related vocal degradation. Little et al. (2008) demonstrated that dysphonia measurements including jitter, shimmer, harmonics-to-noise ratios, and nonlinear dynamic features can discriminate between PD patients and healthy controls with high accuracy.

This project operationalises those findings within a rigorous ML evaluation framework, benchmarking a broad range of classifiers and applying the statistical methodology expected of published clinical ML research.

---

## Dataset

### Primary — UCI Parkinson's Voice Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Authors | Little et al., IEEE Trans Biomed Eng, 2009 |
| Samples | 195 voice recordings |
| Subjects | 31 (23 PD, 8 Healthy) |
| Class Distribution | 147 PD (75.4%), 48 Healthy (24.6%) |
| Features | 22 continuous voice biomarkers |
| Task | Binary classification (PD vs Healthy) |
| Download | https://archive.ics.uci.edu/dataset/174/parkinsons |

### Secondary — UCI Parkinson's Telemonitoring Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Authors | Tsanas et al., IEEE Trans Biomed Eng, 2010 |
| Samples | 5,875 voice recordings |
| Subjects | 42 early-stage PD patients |
| Features | Overlapping voice biomarkers with primary dataset |
| Purpose | Cross-dataset generalization evaluation |
| Download | https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring |

### Feature Categories

| Category | Features | Clinical Meaning |
|---|---|---|
| Frequency | MDVP:Fo(Hz), Fhi(Hz), Flo(Hz) | Stability of vocal fundamental frequency |
| Jitter | MDVP:Jitter(%), Jitter(Abs), RAP, PPQ, DDP | Cycle-to-cycle frequency perturbation |
| Shimmer | MDVP:Shimmer, Shimmer(dB), APQ3, APQ5, APQ11, DDA | Amplitude perturbation |
| Noise | NHR, HNR | Noise-to-harmonics and harmonics-to-noise ratios |
| Nonlinear Dynamics | RPDE, DFA, spread1, spread2, D2, PPE | Recurrence, fractal scaling, and entropy measures |

The application accepts a CSV upload directly through the sidebar. Without a real CSV, a synthetic fallback is generated from the published dataset distributions for demonstration purposes.

---

## System Architecture

```
Parkinsons-Detection-Model/
|
|-- Parkinsons_Detection_App.py     Main application — ten-tab Streamlit dashboard
|-- nn_module.py                    Neural network module — MLP architecture search + PyTorch
|-- advanced_module.py              Advanced ML module — boosting, stacking, statistics
|-- requirements.txt                Full dependency specification
|-- parkinsons.csv                  Real UCI dataset (optional, place here to load automatically)
|-- README.md
```

---

## Models

### Classical Classifiers (scikit-learn)

| Model | Tuning |
|---|---|
| Support Vector Machine (RBF kernel) | GridSearchCV over C and gamma |
| Random Forest | GridSearchCV over n_estimators and max_depth |
| Gradient Boosting | Manual configuration |
| Logistic Regression | L2 regularisation |
| K-Nearest Neighbors | Euclidean distance, k=7 |
| AdaBoost | 100 estimators |
| Gaussian Naive Bayes | Default |
| Decision Tree | max_depth=6 |

### Gradient Boosting Libraries (Optuna-Tuned)

| Model | Optimiser | Trials |
|---|---|---|
| XGBoost | Optuna TPE sampler, 5-fold CV | 40 |
| LightGBM | Optuna TPE sampler, 5-fold CV | 40 |
| CatBoost | Optuna TPE sampler, 5-fold CV | 30 |

Optuna uses the Tree-structured Parzen Estimator (TPE) — a Bayesian optimisation approach that is significantly more sample-efficient than grid or random search.

### Ensemble

| Method | Description |
|---|---|
| Stacking Classifier | Meta-learner (Logistic Regression) trained on out-of-fold predictions from all base models via 5-fold stratified cross-validation |

### Neural Networks

| Model | Configuration |
|---|---|
| MLP (scikit-learn) | Architecture search over 42 configurations — 7 architectures x 2 activation functions x 3 L2 regularisation strengths via GridSearchCV |
| PyTorch Deep NN | 5 architectures from Shallow (64 neurons) to ResBlock-style (256-128-64-32), with BatchNorm, Dropout, Kaiming weight initialisation, ReduceLROnPlateau scheduling, and Early Stopping (patience=20) |

---

## Dashboard — Tab Reference

| Tab | Contents |
|---|---|
| Overview | Model leaderboard ranked by AUC, all ROC curves overlaid, class balance chart, top features by Cohen's d effect size, CV score box distributions |
| Model Performance | Normalised confusion matrix, ROC and Precision-Recall curves, learning curves with bias-variance annotation, decision threshold analysis (F1/Precision/Recall vs cutoff), 10-fold cross-validation fold breakdown |
| Feature Analysis | Random Forest importance by biomarker category, Mann-Whitney U test significance, Cohen's d effect sizes, violin plots with individual data points, PCA 2D class projection, Pearson correlation heatmap |
| Model Comparison | Multi-metric radar chart (six metrics), accuracy vs AUC scatter, grouped bar chart across all metrics and models |
| Hyperparameter Tuning | GridSearchCV results for SVM and Random Forest, before-vs-after ROC curve comparison, AUC improvement quantification |
| SHAP Explainability | Mean absolute SHAP feature importance, beeswarm plot with feature value coloring, dependence plots for top four features, SHAP vs Random Forest importance cross-comparison |
| Neural Networks | MLP architecture search heatmap (architecture x activation), PyTorch training curves (loss, accuracy, learning rate), architecture leaderboard with parameter counts and early stopping epochs, network architecture diagram, NN vs classical ML ROC comparison |
| Advanced ML | Optuna trial histories for XGBoost/LightGBM/CatBoost, stacking ensemble with meta-learner coefficient analysis, SMOTE/ADASYN/SMOTETomek class balance comparison, calibration reliability diagrams (original vs isotonic vs Platt scaling), Bootstrap 95% confidence intervals with forest plot, McNemar's test pairwise matrix, DeLong AUC test pairwise matrix, Wilcoxon signed-rank test, Cohen's d on CV distributions, cross-dataset validation (Voice to Telemonitoring generalization gap) |
| Live Diagnosis | Interactive patient biomarker input with 22 sliders grouped by category, preset patient profiles, single-model prediction with class probabilities, ensemble voting across all models, normalised biomarker deviation chart from healthy reference |
| Data Explorer | Full dataset table with status labels, summary statistics, feature pair scatter with optional KDE density contours, category-wise boxplots |

---

## Methodology

### Data Splitting and Leakage Prevention

All preprocessing (standard scaling) is fit exclusively on the training set and applied as a transform to the test and external validation sets. This prevents data leakage and ensures that reported metrics reflect true out-of-sample performance.

### Validation Protocol

- Stratified 80/20 holdout split with fixed random seed for all primary evaluations
- 10-fold stratified cross-validation for performance estimation and model comparison
- 5-fold stratified cross-validation within hyperparameter search loops
- 1000-iteration bootstrap resampling for confidence interval construction

### Hyperparameter Optimisation

- GridSearchCV is applied to SVM (C, gamma) and Random Forest (n_estimators, max_depth, min_samples_split)
- Optuna TPE Bayesian optimisation is applied to XGBoost, LightGBM, and CatBoost, searching over learning rate, tree depth, subsampling rates, and regularisation parameters
- All optimisation is performed with ROC AUC as the objective to align with the class-imbalanced nature of the dataset

### Class Imbalance Handling

Three resampling strategies are applied and compared:

- SMOTE — generates synthetic minority class samples by interpolation between existing minority samples in feature space
- ADASYN — adaptive version of SMOTE that focuses synthetic generation in regions of higher misclassification density
- SMOTETomek — combined oversampling of minority class and removal of Tomek link boundary samples from majority class

All resampling is applied to training data only. The test set class distribution is preserved intact.

### Model Explainability

- SHAP TreeExplainer provides game-theoretically grounded feature attribution for tree-based models, decomposing each prediction into per-feature contributions
- Beeswarm plots show the distribution of SHAP values across all samples, colored by feature value magnitude
- Dependence plots isolate the marginal effect of individual features on model output across the PD and Healthy subgroups
- Random Forest feature importance (mean decrease in impurity) is provided as a complementary interpretation

### Statistical Significance Testing

| Test | Purpose |
|---|---|
| Mann-Whitney U | Non-parametric test for feature-level group separation between PD and Healthy |
| Cohen's d | Effect size quantification for feature importance ranking |
| McNemar's Test | Pairwise comparison of two classifiers' error patterns on the same test set |
| DeLong's Test | Pairwise comparison of AUC values using variance-covariance estimation of the empirical ROC |
| Wilcoxon Signed-Rank | Non-parametric comparison of paired cross-validation accuracy distributions |
| Bootstrap CI | 95% confidence intervals on AUC via 1000-iteration resampling with replacement |

### Model Calibration

Calibration curves (reliability diagrams) are generated for the top-performing models before and after applying:

- Isotonic regression calibration (non-parametric, more flexible)
- Platt scaling (sigmoid calibration, parametric)

The Brier score is used to quantify calibration quality — lower is better.

### Cross-Dataset Validation

Models trained on the UCI Voice Dataset are evaluated without retraining on the UCI Telemonitoring Dataset, which contains different patients recorded under different conditions. The AUC drop between the in-distribution test set and this external set quantifies the generalization gap and provides a more realistic estimate of deployment performance than standard cross-validation alone.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| ROC AUC | Primary ranking metric — area under the receiver operating characteristic curve |
| PR AUC | Precision-recall AUC — informative under class imbalance |
| F1 Score | Harmonic mean of precision and recall |
| Matthews Correlation Coefficient | Balanced binary classification metric robust to class imbalance |
| Brier Score | Proper scoring rule for probabilistic predictions — measures calibration quality |
| Log Loss | Logarithmic loss for probabilistic classifier evaluation |
| 95% Bootstrap CI | Empirical confidence bounds on AUC |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Clone and Install

```bash
git clone https://github.com/LuxxyJr/Parkinsons-Detection-Model.git
cd Parkinsons-Detection-Model
pip install -r requirements.txt
```

### Run

```bash
streamlit run Parkinsons_Detection_App.py
```

Navigate to http://localhost:8501 in your browser.

### Loading the Real Dataset

Download `parkinsons.csv` from the UCI ML Repository or Kaggle and either:

1. Place it in the project root directory — the application will detect and load it automatically on startup, or
2. Upload it through the sidebar file uploader while the application is running

A green status badge in the application header confirms real data is loaded.

---

## Deployment

### Streamlit Cloud (Recommended — Free)

1. Push all project files to your GitHub repository
2. Visit https://share.streamlit.io and authenticate with GitHub
3. Select New App
4. Configure as follows:
   - Repository: `LuxxyJr/Parkinsons-Detection-Model`
   - Branch: `main`
   - Main file path: `Parkinsons_Detection_App.py`
5. Click Deploy

The application will build and be available at a public URL within approximately two minutes. This URL can be added directly to your CV and portfolio.

---

## Limitations

The following limitations should be acknowledged when interpreting results:

- The UCI Voice Dataset contains 195 samples from 31 subjects. This is a small dataset by contemporary ML standards and results should not be generalised without further validation on larger, independently collected cohorts.
- Multiple recordings per subject introduce within-subject correlation, which standard cross-validation does not fully account for. Subject-level cross-validation would provide more conservative and realistic performance estimates.
- Synthetic telemonitoring data is used for cross-dataset validation demonstrations in the absence of the real dataset file. Download the real UCI Telemonitoring data for production-grade generalization analysis.
- This system has not been evaluated in a clinical setting and is not validated for any diagnostic use.

---

## Disclaimer

This project is developed strictly for educational and research purposes. It is not intended for clinical use, medical diagnosis, treatment planning, or any form of patient management. The predictions produced by this system must not be used as a substitute for professional medical assessment. Parkinson's Disease diagnosis requires evaluation by a qualified neurologist using comprehensive clinical criteria. The authors accept no liability for decisions made on the basis of this system's outputs.

---

## References

1. Little MA, McSharry PE, Hunter EJ, Spielman J, Ramig LO. Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. *IEEE Transactions on Biomedical Engineering*. 2009;56(4):1015–1022.

2. Tsanas A, Little MA, McSharry PE, Ramig LO. Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests. *IEEE Transactions on Biomedical Engineering*. 2010;57(4):884–893.

3. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*. 2017;30.

4. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*. 2002;16:321–357.

5. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*. 1988;44(3):837–845.

6. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2019.

7. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016.

8. Ke G, Meng Q, Finley T, et al. LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*. 2017;30.

9. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*. 2018;31.

---

## License

This project is licensed under the MIT License.

---

## Author

Developed by LuxxyJr as part of a college machine learning research project.

GitHub: https://github.com/LuxxyJr/Parkinsons-Detection-Model
