# 🧠 Parkinson's Disease Detection Lab
**Advanced ML Dashboard — UCI Voice Dataset**

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run parkinsons_app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 What's Inside

### 7 Interactive Tabs

| Tab | What You Get |
|-----|-------------|
| 📡 **Overview** | Model leaderboard, all ROC curves, class balance, effect size rankings |
| 📈 **Model Performance** | Confusion matrix, ROC + PR curves, learning curves, threshold analysis, 10-fold CV |
| 🔬 **Feature Intelligence** | Feature importance, Mann-Whitney U tests, Cohen's d effect sizes, violin plots, PCA, correlation heatmap |
| 🏆 **Model Comparison** | Radar chart, grouped bar chart, scatter plot, all models head-to-head |
| 🩺 **Live Diagnosis** | Slider-based patient input, real-time prediction, ensemble voting, deviation analysis |
| 📊 **Data Explorer** | Raw dataset, pair explorer with KDE contours, category-wise boxplots |
| 🧩 **Explainability** | Permutation importance, partial dependence plots, decision boundary (PCA 2D) |

### 8 Classifiers Trained
- SVM (RBF kernel, C=10)
- Random Forest (200 trees)
- Gradient Boosting (150 estimators)
- Logistic Regression
- K-Nearest Neighbors
- AdaBoost
- Naive Bayes
- Decision Tree

### Statistical Analysis
- **Mann-Whitney U test** — non-parametric significance testing for each feature
- **Cohen's d** — effect size for each biomarker
- **10-Fold Stratified CV** — robust performance estimation
- **Threshold Analysis** — F1/Precision/Recall vs decision threshold

---

## 📊 Dataset
- **Source**: UCI Machine Learning Repository
- **Paper**: Little MA et al., *"Suitability of dysphonia measurements for telemonitoring of Parkinson's disease"*, IEEE Trans Biomed Eng, 2008
- **Samples**: 195 voice recordings (147 PD, 48 Healthy)
- **Features**: 22 voice biomarkers (frequency, jitter, shimmer, noise ratios, nonlinear dynamics)

---

## ⚠️ Disclaimer
This is an **educational and research tool only**.  
It is **not a clinical diagnostic instrument**.  
Always consult a qualified neurologist for medical assessments.
