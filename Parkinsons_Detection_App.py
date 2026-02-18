"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PARKINSON'S DISEASE DETECTION — ADVANCED ML DASHBOARD                     ║
║   Dataset : UCI Parkinson's Voice Dataset (Little et al., 2008)             ║
║   Run     : streamlit run Parkinsons_Detection_App.py                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import warnings, os, io
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    learning_curve, GridSearchCV
)
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# ─── Advanced ML Module ──────────────────────────────────────────────────────
try:
    from advanced_module import render_advanced_tabs
    ADV_MODULE_AVAILABLE = True
except ImportError:
    ADV_MODULE_AVAILABLE = False

# ─── Neural Network Module ────────────────────────────────────────────────────
try:
    from nn_module import render_nn_tab
    NN_MODULE_AVAILABLE = True
except ImportError:
    NN_MODULE_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Parkinson's Detection Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE — Clean Clinical White
# ══════════════════════════════════════════════════════════════════════════════
BG          = "#f4f6f9"
WHITE       = "#ffffff"
NAVY        = "#1e3a5f"
NAVY_LIGHT  = "#2d5282"
BLUE        = "#2563eb"
BLUE_LIGHT  = "#dbeafe"
GREEN       = "#059669"
GREEN_LIGHT = "#d1fae5"
RED         = "#dc2626"
RED_LIGHT   = "#fee2e2"
AMBER       = "#d97706"
AMBER_LIGHT = "#fef3c7"
PURPLE      = "#7c3aed"
PURPLE_LIGHT= "#ede9fe"
SLATE       = "#0ea5e9"
BORDER      = "#e2e8f0"
BORDER_MED  = "#cbd5e1"
TEXT_MAIN   = "#1e293b"
TEXT_MID    = "#475569"
TEXT_DIM    = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor":  WHITE,
    "axes.facecolor":    WHITE,
    "axes.edgecolor":    BORDER_MED,
    "axes.labelcolor":   TEXT_MID,
    "axes.titlecolor":   NAVY,
    "text.color":        TEXT_MAIN,
    "xtick.color":       TEXT_DIM,
    "ytick.color":       TEXT_DIM,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.8,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.facecolor":  WHITE,
    "legend.edgecolor":  BORDER,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

CMAP_MAIN = LinearSegmentedColormap.from_list("m", [GREEN, WHITE, RED], N=256)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #f4f6f9;
    color: #1e293b;
}
div[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    box-shadow: 2px 0 8px rgba(0,0,0,0.05);
}
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-bottom: 2px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600; font-size: 0.82rem;
    color: #94a3b8; background: transparent;
    padding: 10px 22px; border: none;
}
.stTabs [aria-selected="true"] {
    color: #1e3a5f !important;
    border-bottom: 2px solid #1e3a5f !important;
    margin-bottom: -2px;
}
.stButton>button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600; font-size: 0.85rem;
    border: 2px solid #1e3a5f; color: #ffffff;
    background: #1e3a5f; border-radius: 6px;
    padding: 10px 24px; transition: all 0.2s;
}
.stButton>button:hover {
    background: #2d5282; border-color: #2d5282;
    box-shadow: 0 4px 14px rgba(30,58,95,0.3);
    transform: translateY(-1px);
}
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 18px; text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border-left: 4px solid transparent;
}
.card-blue   { border-left-color: #2563eb; }
.card-green  { border-left-color: #059669; }
.card-red    { border-left-color: #dc2626; }
.card-amber  { border-left-color: #d97706; }
.card-purple { border-left-color: #7c3aed; }
.card-navy   { border-left-color: #1e3a5f; }
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; line-height: 1.1; margin-bottom: 4px;
}
.metric-label {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase; color: #94a3b8;
}
.section-title {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase; color: #94a3b8;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px; margin-bottom: 18px;
}
.result-pd {
    background: #fee2e2; border: 2px solid #dc2626;
    border-radius: 12px; padding: 28px; text-align: center;
}
.result-healthy {
    background: #d1fae5; border: 2px solid #059669;
    border-radius: 12px; padding: 28px; text-align: center;
}
.info-box {
    background: #dbeafe; border-left: 4px solid #2563eb;
    padding: 14px 18px; border-radius: 6px;
    font-size: 0.85rem; line-height: 1.75; color: #1e3a5f;
}
.warn-box {
    background: #fef3c7; border-left: 4px solid #d97706;
    padding: 10px 16px; border-radius: 6px;
    font-size: 0.78rem; color: #92400e; font-weight: 500;
}
.success-box {
    background: #d1fae5; border-left: 4px solid #059669;
    padding: 10px 16px; border-radius: 6px;
    font-size: 0.78rem; color: #065f46; font-weight: 500;
}
.badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600;
}
.badge-pd      { background:#fee2e2; color:#dc2626; }
.badge-healthy { background:#d1fae5; color:#059669; }
hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE METADATA
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_META = {
    "MDVP:Fo(Hz)":        dict(desc="Average vocal fundamental frequency",    cat="Frequency", healthy=197.10, pd=145.21, unit="Hz",  lo=80,    hi=280,    fmt=".2f"),
    "MDVP:Fhi(Hz)":       dict(desc="Maximum vocal fundamental frequency",    cat="Frequency", healthy=243.60, pd=197.11, unit="Hz",  lo=90,    hi=400,    fmt=".2f"),
    "MDVP:Flo(Hz)":       dict(desc="Minimum vocal fundamental frequency",    cat="Frequency", healthy=146.67, pd=102.15, unit="Hz",  lo=60,    hi=250,    fmt=".2f"),
    "MDVP:Jitter(%)":     dict(desc="Frequency variation cycle-to-cycle",     cat="Jitter",    healthy=0.0033, pd=0.0063, unit="%",   lo=0.001, hi=0.02,   fmt=".4f"),
    "MDVP:Jitter(Abs)":   dict(desc="Absolute jitter in seconds",             cat="Jitter",    healthy=2.2e-5, pd=4.5e-5, unit="s",   lo=1e-6,  hi=1.5e-4, fmt=".6f"),
    "MDVP:RAP":           dict(desc="Relative amplitude perturbation",        cat="Jitter",    healthy=0.0017, pd=0.0033, unit="",    lo=0.0005,hi=0.012,  fmt=".4f"),
    "MDVP:PPQ":           dict(desc="5-point period perturbation quotient",   cat="Jitter",    healthy=0.0018, pd=0.0034, unit="",    lo=0.0005,hi=0.012,  fmt=".4f"),
    "Jitter:DDP":         dict(desc="Average absolute jitter difference",     cat="Jitter",    healthy=0.0052, pd=0.0100, unit="",    lo=0.001, hi=0.036,  fmt=".4f"),
    "MDVP:Shimmer":       dict(desc="Local shimmer (amplitude variation)",    cat="Shimmer",   healthy=0.0231, pd=0.0508, unit="",    lo=0.008, hi=0.12,   fmt=".4f"),
    "MDVP:Shimmer(dB)":   dict(desc="Shimmer in decibels",                   cat="Shimmer",   healthy=0.214,  pd=0.471,  unit="dB",  lo=0.06,  hi=1.1,    fmt=".3f"),
    "Shimmer:APQ3":       dict(desc="3-point amplitude perturbation quotient",cat="Shimmer",   healthy=0.0122, pd=0.0269, unit="",    lo=0.004, hi=0.065,  fmt=".4f"),
    "Shimmer:APQ5":       dict(desc="5-point amplitude perturbation quotient",cat="Shimmer",   healthy=0.0145, pd=0.0316, unit="",    lo=0.005, hi=0.079,  fmt=".4f"),
    "MDVP:APQ":           dict(desc="11-point amplitude perturbation quotient",cat="Shimmer",  healthy=0.0199, pd=0.0439, unit="",    lo=0.006, hi=0.11,   fmt=".4f"),
    "Shimmer:DDA":        dict(desc="Average absolute shimmer difference",    cat="Shimmer",   healthy=0.0366, pd=0.0808, unit="",    lo=0.012, hi=0.20,   fmt=".4f"),
    "NHR":                dict(desc="Noise-to-harmonics ratio",               cat="Noise",     healthy=0.0111, pd=0.0312, unit="",    lo=0.001, hi=0.30,   fmt=".4f"),
    "HNR":                dict(desc="Harmonics-to-noise ratio",               cat="Noise",     healthy=24.68,  pd=19.98,  unit="dB",  lo=7.0,   hi=34.0,   fmt=".2f"),
    "RPDE":               dict(desc="Recurrence period density entropy",      cat="Nonlinear", healthy=0.499,  pd=0.587,  unit="",    lo=0.25,  hi=0.84,   fmt=".4f"),
    "DFA":                dict(desc="Detrended fluctuation analysis",         cat="Nonlinear", healthy=0.718,  pd=0.753,  unit="",    lo=0.57,  hi=0.88,   fmt=".4f"),
    "spread1":            dict(desc="Nonlinear frequency variation measure",  cat="Nonlinear", healthy=-6.759, pd=-5.335, unit="",    lo=-8.0,  hi=-3.0,   fmt=".3f"),
    "spread2":            dict(desc="Nonlinear frequency variation measure",  cat="Nonlinear", healthy=0.168,  pd=0.269,  unit="",    lo=0.01,  hi=0.52,   fmt=".4f"),
    "D2":                 dict(desc="Correlation dimension",                  cat="Nonlinear", healthy=2.302,  pd=2.522,  unit="",    lo=1.5,   hi=3.5,    fmt=".3f"),
    "PPE":                dict(desc="Pitch period entropy",                   cat="Nonlinear", healthy=0.062,  pd=0.213,  unit="",    lo=0.02,  hi=0.53,   fmt=".4f"),
}
FEAT_NAMES = list(FEATURE_META.keys())
CATS       = ["Frequency", "Jitter", "Shimmer", "Noise", "Nonlinear"]
CAT_COLORS = {"Frequency": BLUE, "Jitter": AMBER, "Shimmer": PURPLE, "Noise": RED, "Nonlinear": GREEN}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (Real UCI CSV  OR  synthetic fallback)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(uploaded=None, seed=42):
    """
    Priority 1 — user uploads parkinsons.csv
    Priority 2 — parkinsons.csv in same directory as script
    Priority 3 — synthetic data faithful to UCI distributions
    """
    # ── Try uploaded file first ──────────────────────────────────────────────
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            if "name" in raw.columns:
                raw = raw.drop(columns=["name"])
            if "status" not in raw.columns:
                st.error("CSV must contain a 'status' column (1=PD, 0=Healthy).")
                return _synthetic(seed), "synthetic"
            df = raw[FEAT_NAMES + ["status"]].dropna()
            return df.reset_index(drop=True), "real"
        except Exception as e:
            st.error(f"Could not read file: {e}")

    # ── Try local file ───────────────────────────────────────────────────────
    local_path = os.path.join(os.path.dirname(__file__), "parkinsons.csv")
    if os.path.exists(local_path):
        try:
            raw = pd.read_csv(local_path)
            if "name" in raw.columns:
                raw = raw.drop(columns=["name"])
            df = raw[FEAT_NAMES + ["status"]].dropna()
            return df.reset_index(drop=True), "real"
        except Exception:
            pass

    # ── Synthetic fallback ───────────────────────────────────────────────────
    return _synthetic(seed), "synthetic"


def _synthetic(seed=42):
    rng = np.random.RandomState(seed)
    n_pd, n_h = 147, 48
    specs = {
        "MDVP:Fo(Hz)":(145.21,30.0,197.10,40.0), "MDVP:Fhi(Hz)":(197.11,55.0,243.60,65.0),
        "MDVP:Flo(Hz)":(102.15,25.0,146.67,35.0), "MDVP:Jitter(%)":(0.0063,0.003,0.0033,0.001),
        "MDVP:Jitter(Abs)":(4.5e-5,2e-5,2.2e-5,8e-6), "MDVP:RAP":(0.0033,0.0016,0.0017,0.0007),
        "MDVP:PPQ":(0.0034,0.0017,0.0018,0.0007), "Jitter:DDP":(0.0100,0.005,0.0052,0.002),
        "MDVP:Shimmer":(0.0508,0.022,0.0231,0.010), "MDVP:Shimmer(dB)":(0.471,0.20,0.214,0.092),
        "Shimmer:APQ3":(0.0269,0.012,0.0122,0.005), "Shimmer:APQ5":(0.0316,0.015,0.0145,0.006),
        "MDVP:APQ":(0.0439,0.020,0.0199,0.009), "Shimmer:DDA":(0.0808,0.036,0.0366,0.015),
        "NHR":(0.0312,0.025,0.0111,0.007), "HNR":(19.98,5.0,24.68,4.0),
        "RPDE":(0.587,0.078,0.499,0.065), "DFA":(0.753,0.044,0.718,0.042),
        "spread1":(-5.335,0.90,-6.759,0.80), "spread2":(0.269,0.095,0.168,0.072),
        "D2":(2.522,0.35,2.302,0.31), "PPE":(0.213,0.10,0.062,0.030),
    }
    rows = {}
    for f, (mu_p,sd_p,mu_h,sd_h) in specs.items():
        m = FEATURE_META[f]
        rows[f] = np.concatenate([
            np.clip(rng.normal(mu_p,sd_p,n_pd), m["lo"], m["hi"]),
            np.clip(rng.normal(mu_h,sd_h,n_h),  m["lo"], m["hi"]),
        ])
    df = pd.DataFrame(rows)
    df["status"] = [1]*n_pd + [0]*n_h
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING + HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_all(df_hash, df):
    X = df[FEAT_NAMES]; y = df["status"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ── GridSearchCV on best candidates ─────────────────────────────────────
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        {"C": [1, 10, 50], "gamma": [0.001, 0.01, 0.1], "kernel": ["rbf"]},
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc", n_jobs=-1
    )
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [6, 10, None], "min_samples_split": [2, 4]},
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc", n_jobs=-1
    )
    svm_grid.fit(Xtr, ytr)
    rf_grid.fit(Xtr, ytr)
    best_svm = svm_grid.best_estimator_
    best_rf  = rf_grid.best_estimator_

    model_zoo = {
        "SVM (Tuned)":          best_svm,
        "Random Forest (Tuned)":best_rf,
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
        "Logistic Regression":  LogisticRegression(C=0.5, max_iter=2000, random_state=42),
        "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=7),
        "AdaBoost":             AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        "Naive Bayes":          GaussianNB(),
        "Decision Tree":        DecisionTreeClassifier(max_depth=6, min_samples_split=5, random_state=42),
    }

    res = {}
    for name, clf in model_zoo.items():
        if name not in ["SVM (Tuned)", "Random Forest (Tuned)"]:
            clf.fit(Xtr, ytr)
        yp  = clf.predict(Xte)
        ypr = clf.predict_proba(Xte)[:,1]
        cvs = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy")
        fpr, tpr, _ = roc_curve(yte, ypr)
        pr_p, pr_r, _ = precision_recall_curve(yte, ypr)
        res[name] = dict(
            clf=clf, accuracy=accuracy_score(yte,yp),
            f1=f1_score(yte,yp), precision=precision_score(yte,yp),
            recall=recall_score(yte,yp), mcc=matthews_corrcoef(yte,yp),
            roc_auc=auc(fpr,tpr), pr_auc=average_precision_score(yte,ypr),
            cv_mean=cvs.mean(), cv_std=cvs.std(), cv_all=cvs,
            cm=confusion_matrix(yte,yp),
            report=classification_report(yte,yp, target_names=["Healthy","PD"]),
            fpr=fpr, tpr=tpr, pr_p=pr_p, pr_r=pr_r,
            y_test=yte, y_pred=yp, y_proba=ypr,
        )

    tuning_info = {
        "SVM (Tuned)":          {"best_params": svm_grid.best_params_, "best_cv_auc": svm_grid.best_score_},
        "Random Forest (Tuned)":{"best_params": rf_grid.best_params_,  "best_cv_auc": rf_grid.best_score_},
    }
    return res, scaler, Xtr, Xte, ytr, yte, tuning_info


@st.cache_data
def compute_stat_tests(df_hash, df):
    tests = {}
    for f in FEAT_NAMES:
        pd_v = df[df.status==1][f]; h_v = df[df.status==0][f]
        u, p = stats.mannwhitneyu(pd_v, h_v, alternative="two-sided")
        d    = (pd_v.mean()-h_v.mean()) / np.sqrt((pd_v.std()**2+h_v.std()**2)/2)
        tests[f] = dict(p=p, d=abs(d), u=u)
    return tests


@st.cache_data
def compute_shap(df_hash, df, _scaler):
    if not SHAP_AVAILABLE:
        return None, None
    X  = df[FEAT_NAMES]; y = df["status"]
    Xs = _scaler.transform(X)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(Xs, y)
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xs)
    # For binary: shap_values is list [class0, class1] — take class 1 (PD)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    return sv, Xs


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#1e3a5f;margin-bottom:2px'>🧠 PD Detection</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.65rem;color:#94a3b8;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:20px'>ML Research Dashboard</div>", unsafe_allow_html=True)

    # ── Dataset upload ───────────────────────────────────────────────────────
    st.markdown("**Dataset**")
    uploaded_file = st.file_uploader("Upload parkinsons.csv", type=["csv"], label_visibility="collapsed")
    st.markdown("""<div class="info-box" style="font-size:0.72rem;padding:10px 14px">
        <strong>📥 Real Dataset</strong><br>
        Download <code>parkinsons.csv</code> from
        <a href="https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set" target="_blank">Kaggle</a>
        or <a href="https://archive.ics.uci.edu/dataset/174/parkinsons" target="_blank">UCI</a>
        and upload it here for real data analysis.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Select Classifier**")
    # placeholder — filled after training
    model_placeholder = st.empty()

    st.markdown("---")
    st.markdown("<div style='font-size:0.65rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#94a3b8;margin-bottom:8px'>Dataset Info</div>", unsafe_allow_html=True)
    dataset_info_placeholder = st.empty()

    st.markdown("---")
    st.markdown("""<div class="warn-box">
        ⚠️ Research & educational use only.<br>
        Not a clinical diagnostic tool.<br>
        Always consult a neurologist.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA  (depends on sidebar upload)
# ══════════════════════════════════════════════════════════════════════════════
df, data_source = load_data(uploaded_file)
df_hash = len(df)  # simple cache key

results, scaler, Xtr, Xte, ytr, yte, tuning_info = train_all(df_hash, df)
stat_tests = compute_stat_tests(df_hash, df)

# SHAP (lazy — only compute when tab opened)
shap_vals_cache = {}

# Fill sidebar model selector
with model_placeholder:
    selected_model = st.selectbox("Classifier", list(results.keys()), label_visibility="collapsed")
r = results[selected_model]

# Fill sidebar dataset info
n_pd = int((df.status==1).sum()); n_h = int((df.status==0).sum())
with dataset_info_placeholder:
    info_rows = [
        ("Source",   "UCI ML Repository"),
        ("Status",   "✅ Real Data" if data_source=="real" else "⚠️ Synthetic"),
        ("Samples",  str(len(df))),
        ("PD",       f"{n_pd} ({n_pd/len(df)*100:.1f}%)"),
        ("Healthy",  f"{n_h} ({n_h/len(df)*100:.1f}%)"),
        ("Features", f"{len(FEAT_NAMES)} biomarkers"),
        ("Models",   f"{len(results)} classifiers"),
    ]
    for k, v in info_rows:
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.78rem;margin:4px 0'><span style='color:#94a3b8'>{k}</span><span style='color:#475569'>{v}</span></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
data_badge = f"<span style='background:#d1fae5;color:#065f46;padding:3px 12px;border-radius:20px;font-size:0.72rem;font-weight:600'>✅ Real UCI Data</span>" if data_source=="real" else \
             f"<span style='background:#fef3c7;color:#92400e;padding:3px 12px;border-radius:20px;font-size:0.72rem;font-weight:600'>⚠️ Synthetic Data — Upload CSV for real results</span>"

st.markdown(f"""
<div style='padding:24px 0 16px 0;border-bottom:2px solid #e2e8f0;margin-bottom:24px'>
  <div style='font-family:DM Serif Display,serif;font-size:2.6rem;color:#1e3a5f;line-height:1.1'>
    Parkinson's Detection Lab
  </div>
  <div style='font-size:0.82rem;color:#94a3b8;margin-top:8px'>
    Voice &amp; Gait Biomarker Analysis &nbsp;·&nbsp; UCI ML Repository &nbsp;·&nbsp;
    Little et al. 2008 &nbsp;·&nbsp; 22 Features &nbsp;·&nbsp; 8 Classifiers
    &nbsp;&nbsp;{data_badge}
  </div>
</div>
""", unsafe_allow_html=True)

# Top metric strip
cols = st.columns(6)
cards = [
    ("ACCURACY",   f"{r['accuracy']*100:.1f}%",  "blue"),
    ("F1 SCORE",   f"{r['f1']:.4f}",              "green"),
    ("ROC AUC",    f"{r['roc_auc']:.4f}",         "amber"),
    ("PRECISION",  f"{r['precision']:.4f}",        "purple"),
    ("RECALL",     f"{r['recall']:.4f}",           "red"),
    ("10-FOLD CV", f"{r['cv_mean']*100:.1f}±{r['cv_std']*100:.1f}%", "navy"),
]
colors_map = dict(blue=BLUE, green=GREEN, amber=AMBER, purple=PURPLE, red=RED, navy=NAVY)
for col, (lbl, val, color) in zip(cols, cards):
    col.markdown(f'<div class="card card-{color}"><div class="metric-val" style="color:{colors_map[color]}">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📡 Overview",
    "📈 Model Performance",
    "🔬 Feature Analysis",
    "🏆 Model Comparison",
    "⚙️ Hyperparameter Tuning",
    "🧩 SHAP Explainability",
    "🤖 Neural Networks",
    "🔥 Advanced ML",
    "🩺 Live Diagnosis",
    "📊 Data Explorer",
])


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 0 — OVERVIEW
# └─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    pal = [BLUE, GREEN, AMBER, PURPLE, RED, SLATE, "#f43f5e", "#84cc16"]

    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown('<div class="section-title">Model Leaderboard</div>', unsafe_allow_html=True)
        lb_rows = []
        for name, res in results.items():
            tuned = "✅" if name in tuning_info else "—"
            lb_rows.append({
                "Model": name, "Tuned": tuned,
                "Accuracy": f"{res['accuracy']*100:.2f}%",
                "F1": f"{res['f1']:.4f}", "AUC": f"{res['roc_auc']:.4f}",
                "PR-AUC": f"{res['pr_auc']:.4f}", "MCC": f"{res['mcc']:.4f}",
                "10-CV Acc": f"{res['cv_mean']*100:.1f}±{res['cv_std']*100:.1f}%",
            })
        lb_df = pd.DataFrame(lb_rows).sort_values("AUC", ascending=False).reset_index(drop=True)
        lb_df.index += 1
        st.dataframe(lb_df, width="stretch")

        st.markdown('<div class="section-title" style="margin-top:24px">ROC Curves — All Models</div>', unsafe_allow_html=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for (name, res), c in zip(results.items(), pal):
            ax1.plot(res["fpr"], res["tpr"], lw=2, color=c, label=f"{name[:22]} ({res['roc_auc']:.3f})")
            ax2.plot(res["pr_r"], res["pr_p"], lw=2, color=c, label=name[:22])
        ax1.plot([0,1],[0,1], color=BORDER_MED, lw=1, ls="--", label="Random baseline")
        ax1.set(title="ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax1.legend(fontsize=6.5); ax1.grid(True)
        ax2.set(title="Precision-Recall Curves", xlabel="Recall", ylabel="Precision")
        ax2.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c2:
        st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        wedges, _, at = ax.pie(
            [n_h, n_pd], labels=["Healthy", "PD"],
            colors=[GREEN, RED], autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor=WHITE, linewidth=2.5, width=0.55)
        )
        for t in at: t.set_color(WHITE); t.set_fontsize(11); t.set_fontweight("bold")
        ax.set_title("Dataset Balance", pad=10)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown('<div class="section-title" style="margin-top:16px">Top Features by Effect Size</div>', unsafe_allow_html=True)
        top5 = sorted(stat_tests.items(), key=lambda x: -x[1]["d"])[:6]
        for f, st_res in top5:
            d = st_res["d"]; p = st_res["p"]
            sig = "***" if p<0.001 else ("**" if p<0.01 else "*")
            bar_w = min(d/3.0, 1.0)
            st.markdown(f"""<div style='margin:8px 0'>
              <div style='display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:3px'>
                <span style='color:{TEXT_MID};font-weight:500'>{f}</span>
                <span style='color:{AMBER};font-weight:700'>d={d:.2f} {sig}</span>
              </div>
              <div style='height:6px;background:{BORDER};border-radius:3px;overflow:hidden'>
                <div style='height:100%;width:{bar_w*100:.0f}%;background:{AMBER};border-radius:3px;opacity:0.8'></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:20px">CV Score Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cv_data  = [results[n]["cv_all"]*100 for n in results]
        cv_names = [n.replace(" ","\n").replace("(","(\n") for n in results]
        bp = ax.boxplot(cv_data, labels=cv_names, patch_artist=True, widths=0.5,
                        medianprops=dict(color=NAVY, lw=2),
                        whiskerprops=dict(color=TEXT_DIM),
                        capprops=dict(color=TEXT_DIM),
                        flierprops=dict(marker="o", ms=3, alpha=0.5, markerfacecolor=RED))
        for patch, c in zip(bp["boxes"], pal):
            patch.set(facecolor=c+"22", edgecolor=c, linewidth=1.5)
        ax.set(title="10-Fold CV Accuracy (%)", ylabel="Accuracy (%)"); ax.grid(axis="y")
        plt.xticks(fontsize=6); fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 1 — MODEL PERFORMANCE
# └─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    r = results[selected_model]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        cm_n = r["cm"].astype(float) / r["cm"].sum(axis=1, keepdims=True)
        im = ax.imshow(cm_n, cmap=LinearSegmentedColormap.from_list("cm",[WHITE,BLUE_LIGHT,BLUE],256), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{r['cm'][i,j]}\n({cm_n[i,j]*100:.1f}%)",
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color=WHITE if cm_n[i,j]>0.6 else NAVY)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred: Healthy","Pred: PD"], fontsize=9)
        ax.set_yticklabels(["True: Healthy","True: PD"], fontsize=9)
        ax.set_title(f"{selected_model} — Confusion Matrix", pad=12)
        plt.colorbar(im, ax=ax, fraction=0.046, label="Proportion")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown('<div class="section-title" style="margin-top:16px">Classification Report</div>', unsafe_allow_html=True)
        st.code(r["report"], language="")

    with col2:
        st.markdown('<div class="section-title">ROC + Precision-Recall</div>', unsafe_allow_html=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
        ax1.plot(r["fpr"], r["tpr"], color=BLUE, lw=2.5, label=f"AUC = {r['roc_auc']:.4f}")
        ax1.fill_between(r["fpr"], r["tpr"], alpha=0.08, color=BLUE)
        ax1.plot([0,1],[0,1], color=BORDER_MED, lw=1.2, ls="--", label="Random (0.5)")
        ax1.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax1.legend(); ax1.grid(True)
        ax2.plot(r["pr_r"], r["pr_p"], color=GREEN, lw=2.5, label=f"AP = {r['pr_auc']:.4f}")
        ax2.fill_between(r["pr_r"], r["pr_p"], alpha=0.08, color=GREEN)
        ax2.axhline(n_pd/len(df), color=BORDER_MED, lw=1.2, ls="--", label="Baseline")
        ax2.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision", ylim=[0,1.05])
        ax2.legend(); ax2.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title" style="margin-top:8px">Learning Curves</div>', unsafe_allow_html=True)
    Xs_full = scaler.transform(df[FEAT_NAMES])
    with st.spinner("Computing learning curves…"):
        tr_sz, tr_sc, val_sc = learning_curve(
            r["clf"], Xs_full, df["status"],
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring="accuracy", n_jobs=-1
        )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(tr_sz, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.12, color=BLUE)
    ax.fill_between(tr_sz, val_sc.mean(1)-val_sc.std(1), val_sc.mean(1)+val_sc.std(1), alpha=0.12, color=GREEN)
    ax.plot(tr_sz, tr_sc.mean(1), "o-", color=BLUE,  lw=2, ms=5, label="Training Score")
    ax.plot(tr_sz, val_sc.mean(1), "s-", color=GREEN, lw=2, ms=5, label="Cross-Val Score")
    gap = tr_sc.mean(1)[-1] - val_sc.mean(1)[-1]
    ax.set(title=f"{selected_model} — Learning Curves (Bias-Variance Gap: {gap:.3f})",
           xlabel="Training Samples", ylabel="Accuracy")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Decision Threshold Analysis</div>', unsafe_allow_html=True)
        thresholds = np.linspace(0.01, 0.99, 99)
        f1s, precs, recs, accs = [], [], [], []
        for t in thresholds:
            yp_t = (r["y_proba"] >= t).astype(int)
            f1s.append(f1_score(r["y_test"], yp_t, zero_division=0))
            precs.append(precision_score(r["y_test"], yp_t, zero_division=0))
            recs.append(recall_score(r["y_test"], yp_t, zero_division=0))
            accs.append(accuracy_score(r["y_test"], yp_t))
        best_t = thresholds[np.argmax(f1s)]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(thresholds, f1s,   color=BLUE,   lw=2, label="F1")
        ax.plot(thresholds, precs, color=GREEN,  lw=2, label="Precision")
        ax.plot(thresholds, recs,  color=AMBER,  lw=2, label="Recall")
        ax.plot(thresholds, accs,  color=PURPLE, lw=2, label="Accuracy", ls="--")
        ax.axvline(best_t, color=RED, lw=1.5, ls=":", label=f"Best F1 @ {best_t:.2f}")
        ax.set(title="Metrics vs Decision Threshold", xlabel="Threshold", ylabel="Score", ylim=[0,1.05])
        ax.legend(fontsize=7.5); ax.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">10-Fold Cross-Validation</div>', unsafe_allow_html=True)
        cvs = r["cv_all"] * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bar_colors = [GREEN if v >= cvs.mean() else AMBER for v in cvs]
        bars = ax.bar(range(1,11), cvs, color=bar_colors, alpha=0.8, edgecolor=WHITE, zorder=2)
        ax.axhline(cvs.mean(), color=NAVY, lw=2, ls="--", label=f"Mean: {cvs.mean():.2f}%", zorder=3)
        ax.axhline(cvs.mean()-cvs.std(), color=NAVY, lw=1, ls=":", alpha=0.5, zorder=3)
        ax.axhline(cvs.mean()+cvs.std(), color=NAVY, lw=1, ls=":", alpha=0.5, label=f"±1σ: {cvs.std():.2f}%", zorder=3)
        for bar, v in zip(bars, cvs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_MID)
        ax.set(title="10-Fold CV Accuracy", xlabel="Fold", ylabel="Accuracy (%)", ylim=[60,105])
        ax.legend(fontsize=8); ax.grid(axis="y", zorder=0)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 2 — FEATURE ANALYSIS
# └─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    rf_res = results["Random Forest (Tuned)"]
    imp    = rf_res["clf"].feature_importances_
    idx    = np.argsort(imp)[::-1]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 7))
        top_n = 18; feat_top = [FEAT_NAMES[i] for i in idx[:top_n]]; imp_top = imp[idx[:top_n]]
        cats_top = [FEATURE_META[f]["cat"] for f in feat_top]
        bar_colors = [CAT_COLORS[c] for c in cats_top]
        ax.barh(feat_top[::-1], imp_top[::-1], color=bar_colors[::-1], alpha=0.82, edgecolor=WHITE, height=0.7)
        ax.set(title="Top 18 Feature Importances", xlabel="Mean Decrease in Impurity")
        ax.grid(axis="x")
        handles = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CATS]
        ax.legend(handles=handles, loc="lower right", fontsize=7.5)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">Statistical Significance (Mann-Whitney U)</div>', unsafe_allow_html=True)
        st_sorted = sorted(stat_tests.items(), key=lambda x: -x[1]["d"])
        feat_sig  = [f for f,_ in st_sorted]; d_vals=[v["d"] for _,v in st_sorted]; p_vals=[v["p"] for _,v in st_sorted]
        sig_colors= [RED if p<0.001 else (AMBER if p<0.01 else (GREEN if p<0.05 else TEXT_DIM)) for p in p_vals]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 7))
        ax1.barh(feat_sig[::-1], d_vals[::-1], color=sig_colors[::-1], alpha=0.8, edgecolor=WHITE, height=0.7)
        ax1.axvline(0.5, color=AMBER, lw=1, ls="--", label="Medium (0.5)")
        ax1.axvline(0.8, color=RED,   lw=1, ls="--", label="Large (0.8)")
        ax1.set(title="Cohen's d Effect Size", xlabel="|d|"); ax1.legend(fontsize=6.5); ax1.grid(axis="x")
        neg_log_p = [-np.log10(max(p,1e-10)) for p in p_vals]
        bar_colors2 = [RED if v>=-np.log10(0.001) else (AMBER if v>=-np.log10(0.01) else (GREEN if v>=-np.log10(0.05) else TEXT_DIM)) for v in neg_log_p]
        ax2.barh(feat_sig[::-1], neg_log_p[::-1], color=bar_colors2[::-1], alpha=0.8, edgecolor=WHITE, height=0.7)
        ax2.axvline(-np.log10(0.001), color=RED,   lw=1, ls="--", label="p=0.001")
        ax2.axvline(-np.log10(0.05),  color=GREEN, lw=1, ls="--", label="p=0.05")
        ax2.set(title="-log₁₀(p-value)"); ax2.legend(fontsize=6.5); ax2.grid(axis="x"); ax2.set_yticklabels([])
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title" style="margin-top:4px">Violin Plots — Top 8 Features</div>', unsafe_allow_html=True)
    top8 = [FEAT_NAMES[i] for i in idx[:8]]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for i, (feat, ax) in enumerate(zip(top8, axes.flatten())):
        pd_v = df[df.status==1][feat].values; h_v = df[df.status==0][feat].values
        parts = ax.violinplot([h_v, pd_v], positions=[0,1], showmedians=True)
        for pc, color in zip(parts["bodies"], [GREEN, RED]):
            pc.set(facecolor=color, alpha=0.35, edgecolor=color)
        parts["cmedians"].set(color=NAVY, lw=2)
        for key in ["cbars","cmins","cmaxes"]: parts[key].set(color=TEXT_DIM, lw=1)
        ax.scatter([0]*len(h_v),  h_v,  color=GREEN, alpha=0.25, s=6, zorder=3)
        ax.scatter([1]*len(pd_v), pd_v, color=RED,   alpha=0.25, s=6, zorder=3)
        d=stat_tests[feat]["d"]; p=stat_tests[feat]["p"]
        sig="***" if p<0.001 else "**" if p<0.01 else "*"
        ax.set_title(f"{feat}\nd={d:.2f} {sig}", fontsize=7.5, pad=4)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Healthy","PD"], fontsize=8); ax.grid(axis="y")
        for spine in ax.spines.values(): spine.set_edgecolor(CAT_COLORS[FEATURE_META[feat]["cat"]]); spine.set_linewidth(1.5)
    fig.suptitle("Feature Distributions — PD vs Healthy", y=1.01, color=NAVY, fontsize=11, fontweight="bold")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">PCA Visualization</div>', unsafe_allow_html=True)
        pca2 = PCA(n_components=2, random_state=42)
        Xp   = pca2.fit_transform(scaler.transform(df[FEAT_NAMES]))
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for label, color, name in [(0,GREEN,"Healthy"),(1,RED,"PD")]:
            mask = df.status==label
            ax.scatter(Xp[mask,0], Xp[mask,1], c=color, alpha=0.55, s=30,
                       edgecolors=WHITE, lw=0.5, label=f"{name} (n={mask.sum()})", zorder=3)
        ax.set(title=f"PCA — PC1:{pca2.explained_variance_ratio_[0]*100:.1f}%  PC2:{pca2.explained_variance_ratio_[1]*100:.1f}%",
               xlabel="PC1", ylabel="PC2"); ax.legend(); ax.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">Correlation Heatmap (Top 12)</div>', unsafe_allow_html=True)
        top12 = [FEAT_NAMES[i] for i in idx[:12]]
        corr  = df[top12+["status"]].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f",
                    cmap=LinearSegmentedColormap.from_list("div",[GREEN,WHITE,RED],256),
                    ax=ax, linewidths=0.4, linecolor=BORDER,
                    annot_kws={"size":6.5}, vmin=-1, vmax=1,
                    cbar_kws={"shrink":0.7})
        ax.set_title("Feature Correlation Matrix", pad=10)
        plt.xticks(rotation=45, ha="right", fontsize=7); plt.yticks(fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 3 — MODEL COMPARISON
# └─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    pal = [BLUE, GREEN, AMBER, PURPLE, RED, SLATE, "#f43f5e", "#84cc16"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">ROC Curves</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, res), c in zip(results.items(), pal):
            ax.plot(res["fpr"], res["tpr"], lw=2, color=c, label=f"{name} ({res['roc_auc']:.3f})")
        ax.plot([0,1],[0,1], color=BORDER_MED, lw=1, ls="--")
        ax.set(title="ROC Curves — All Classifiers", xlabel="FPR", ylabel="TPR")
        ax.legend(loc="lower right", fontsize=7); ax.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">Accuracy vs AUC</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, res), c in zip(results.items(), pal):
            ax.scatter(res["accuracy"]*100, res["roc_auc"], color=c, s=130,
                       edgecolors=BORDER_MED, lw=1.5, zorder=4)
            ax.annotate(name, (res["accuracy"]*100, res["roc_auc"]),
                        textcoords="offset points", xytext=(8,4), fontsize=7, color=c)
        ax.set(title="Accuracy vs ROC-AUC", xlabel="Accuracy (%)", ylabel="ROC AUC")
        ax.grid(True); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title">Multi-Metric Radar</div>', unsafe_allow_html=True)
    metrics_r = ["accuracy","f1","roc_auc","pr_auc","recall","precision"]
    labels_r  = ["Accuracy","F1","ROC-AUC","PR-AUC","Recall","Precision"]
    N = len(metrics_r)
    angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8,5.5), subplot_kw=dict(polar=True))
    ax.set_facecolor(BG)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_r, size=9, color=TEXT_MID)
    ax.set_ylim(0.5, 1.0); ax.set_yticks([0.6,0.7,0.8,0.9,1.0])
    ax.set_yticklabels(["60","70","80","90","100"], size=7, color=TEXT_DIM); ax.grid(color=BORDER, lw=0.8)
    for (name, res), c in zip(results.items(), pal):
        vals = [res[m] for m in metrics_r]+[res[metrics_r[0]]]
        ax.plot(angles, vals, lw=2, color=c, label=name)
        ax.fill(angles, vals, alpha=0.06, color=c)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.15), fontsize=7.5)
    ax.set_title("Multi-Metric Comparison — All Models", pad=20, color=NAVY, fontweight="bold")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title">All Models × All Metrics</div>', unsafe_allow_html=True)
    names_b = list(results.keys())
    metric_keys   = ["accuracy","f1","roc_auc","pr_auc","mcc"]
    metric_labels = ["Accuracy","F1 Score","ROC AUC","PR AUC","MCC"]
    x = np.arange(len(names_b)); w = 0.15
    fig, ax = plt.subplots(figsize=(13, 4))
    for i,(mk,ml,c) in enumerate(zip(metric_keys, metric_labels, [BLUE,GREEN,AMBER,PURPLE,RED])):
        ax.bar(x+i*w, [results[n][mk] for n in names_b], w, label=ml, color=c, alpha=0.8, edgecolor=WHITE)
    ax.set(title="All Models × All Metrics", ylabel="Score", xticks=x+w*2)
    ax.set_xticklabels([n.replace(" ","\n") for n in names_b], fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis="y"); ax.set_ylim(0,1.1)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 4 — HYPERPARAMETER TUNING
# └─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("""<div class="info-box">
        <strong>GridSearchCV</strong> was applied to the two strongest models — <strong>SVM</strong> and
        <strong>Random Forest</strong> — using 5-fold stratified cross-validation optimising for
        <strong>ROC AUC</strong>. Results below show the best hyperparameters found and their impact.
    </div><br>""", unsafe_allow_html=True)

    for model_name in ["SVM (Tuned)", "Random Forest (Tuned)"]:
        info = tuning_info[model_name]
        res  = results[model_name]
        st.markdown(f'<div class="section-title">{model_name}</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"""<div class="card card-navy">
            <div class="metric-val" style="color:{NAVY};font-size:1.4rem">{str(info['best_params'])}</div>
            <div class="metric-label">Best Parameters</div>
        </div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="card card-blue">
            <div class="metric-val" style="color:{BLUE}">{info['best_cv_auc']:.4f}</div>
            <div class="metric-label">Best CV ROC AUC</div>
        </div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div class="card card-green">
            <div class="metric-val" style="color:{GREEN}">{res['accuracy']*100:.1f}%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Before vs After Tuning — SVM</div>', unsafe_allow_html=True)
    # Show default SVM vs tuned SVM comparison
    Xs = scaler.transform(df[FEAT_NAMES])
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xs, df["status"], test_size=0.2, stratify=df["status"], random_state=42)

    default_svm = SVC(kernel="rbf", probability=True, random_state=42)
    default_svm.fit(Xtr2, ytr2)
    default_proba = default_svm.predict_proba(Xte2)[:,1]
    default_fpr, default_tpr, _ = roc_curve(yte2, default_proba)
    default_auc_val = auc(default_fpr, default_tpr)

    tuned_fpr = results["SVM (Tuned)"]["fpr"]
    tuned_tpr = results["SVM (Tuned)"]["tpr"]
    tuned_auc_val = results["SVM (Tuned)"]["roc_auc"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(default_fpr, default_tpr, color=AMBER, lw=2, ls="--", label=f"Default SVM (AUC={default_auc_val:.4f})")
    ax.plot(tuned_fpr,   tuned_tpr,   color=BLUE,  lw=2.5,         label=f"Tuned SVM   (AUC={tuned_auc_val:.4f})")
    ax.fill_between(tuned_fpr, tuned_tpr, alpha=0.08, color=BLUE)
    ax.plot([0,1],[0,1], color=BORDER_MED, lw=1, ls=":")
    ax.set(title="SVM — Default vs GridSearchCV Tuned", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    improvement = (tuned_auc_val - default_auc_val)*100
    st.markdown(f"""<div class="{'success-box' if improvement>0 else 'warn-box'}">
        AUC improvement from tuning: <strong>{improvement:+.2f}%</strong> &nbsp;
        (Default: {default_auc_val:.4f} → Tuned: {tuned_auc_val:.4f})
    </div>""", unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 5 — SHAP EXPLAINABILITY
# └─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    if not SHAP_AVAILABLE:
        st.markdown("""<div class="warn-box">
            <strong>SHAP not installed.</strong> Run <code>pip install shap</code> then restart the app.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="info-box">
            <strong>SHAP (SHapley Additive exPlanations)</strong> is the industry-standard method for
            explaining ML model predictions. It assigns each feature an importance value for a specific
            prediction based on game theory. Unlike simple feature importance, SHAP shows both the
            <em>magnitude</em> and <em>direction</em> of each feature's contribution.
        </div><br>""", unsafe_allow_html=True)

        with st.spinner("Computing SHAP values (this takes ~15 seconds)…"):
            sv, Xs_shap = compute_shap(df_hash, df, scaler)

        if sv is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-title">SHAP Feature Importance (Mean |SHAP|)</div>', unsafe_allow_html=True)
                mean_shap = np.abs(sv).mean(axis=0)
                shap_idx  = np.argsort(mean_shap)[::-1]
                fig, ax = plt.subplots(figsize=(6, 7))
                top15_feat = [FEAT_NAMES[i] for i in shap_idx[:15]]
                top15_shap = mean_shap[shap_idx[:15]]
                bar_colors = [CAT_COLORS[FEATURE_META[f]["cat"]] for f in top15_feat]
                ax.barh(top15_feat[::-1], top15_shap[::-1], color=bar_colors[::-1], alpha=0.82, edgecolor=WHITE, height=0.7)
                ax.set(title="Mean |SHAP Value| — Top 15 Features\n(Higher = More Important to Model)",
                       xlabel="Mean |SHAP|")
                ax.grid(axis="x")
                handles = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CATS]
                ax.legend(handles=handles, loc="lower right", fontsize=7.5)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            with col2:
                st.markdown('<div class="section-title">SHAP Beeswarm Plot</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 7))
                top10_idx = shap_idx[:10]
                top10_feat = [FEAT_NAMES[i] for i in top10_idx]
                top10_sv   = sv[:, top10_idx]
                top10_Xs   = Xs_shap[:, top10_idx]

                for j, (feat, feat_idx) in enumerate(zip(top10_feat[::-1], top10_idx[::-1])):
                    sv_feat   = sv[:, feat_idx]
                    val_feat  = Xs_shap[:, feat_idx]
                    # Normalise values for coloring
                    val_norm  = (val_feat - val_feat.min()) / (val_feat.max() - val_feat.min() + 1e-9)
                    colors    = plt.cm.RdYlGn(1 - val_norm)  # red=high, green=low
                    y_jitter  = j + np.random.uniform(-0.18, 0.18, len(sv_feat))
                    sc = ax.scatter(sv_feat, y_jitter, c=val_norm, cmap="RdYlGn_r",
                                   s=12, alpha=0.6, zorder=3)

                ax.set_yticks(range(len(top10_feat)))
                ax.set_yticklabels(top10_feat[::-1], fontsize=8)
                ax.axvline(0, color=NAVY, lw=1.5, ls="--")
                ax.set(title="SHAP Beeswarm — Top 10 Features\n(Right=Increases PD risk, Left=Decreases)",
                       xlabel="SHAP Value (impact on PD probability)")
                ax.grid(axis="x")
                plt.colorbar(sc, ax=ax, label="Feature value (low→green, high→red)", shrink=0.6)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.markdown('<div class="section-title">SHAP Dependence Plots — Top 4 Features</div>', unsafe_allow_html=True)
            top4 = [FEAT_NAMES[i] for i in shap_idx[:4]]
            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            for ax, feat in zip(axes, top4):
                feat_i   = FEAT_NAMES.index(feat)
                sv_feat  = sv[:, feat_i]
                val_feat = Xs_shap[:, feat_i]
                status   = df["status"].values
                ax.scatter(val_feat[status==0], sv_feat[status==0], color=GREEN, alpha=0.5, s=15, label="Healthy", zorder=3)
                ax.scatter(val_feat[status==1], sv_feat[status==1], color=RED,   alpha=0.5, s=15, label="PD",      zorder=3)
                ax.axhline(0, color=NAVY, lw=1, ls="--")
                ax.set(title=feat, xlabel="Feature Value (scaled)", ylabel="SHAP Value")
                ax.legend(fontsize=7); ax.grid(True)
            fig.suptitle("SHAP Dependence Plots — How each feature drives PD prediction",
                         y=1.02, color=NAVY, fontsize=11, fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.markdown('<div class="section-title">SHAP vs RF Feature Importance Comparison</div>', unsafe_allow_html=True)
            rf_imp = results["Random Forest (Tuned)"]["clf"].feature_importances_
            fig, ax = plt.subplots(figsize=(12, 3.5))
            x   = np.arange(len(FEAT_NAMES)); w = 0.38
            rf_norm   = rf_imp / rf_imp.max()
            shap_norm = mean_shap / mean_shap.max()
            ax.bar(x-w/2, rf_norm,   w, label="RF Importance (normalized)", color=BLUE,  alpha=0.75, edgecolor=WHITE)
            ax.bar(x+w/2, shap_norm, w, label="SHAP Importance (normalized)", color=AMBER, alpha=0.75, edgecolor=WHITE)
            ax.set_xticks(x); ax.set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=7)
            ax.set(title="RF Feature Importance vs SHAP — Normalized Comparison", ylabel="Normalized Importance")
            ax.legend(fontsize=8.5); ax.grid(axis="y")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 6 — NEURAL NETWORKS
# └─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    if NN_MODULE_AVAILABLE:
        render_nn_tab(df, scaler, FEAT_NAMES, results)
    else:
        st.markdown("""<div class="warn-box">
            <strong>nn_module.py not found.</strong> Make sure <code>nn_module.py</code>
            is in the same folder as <code>Parkinsons_Detection_App.py</code>.
        </div>""", unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 7 — ADVANCED ML
# └─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    if ADV_MODULE_AVAILABLE:
        render_advanced_tabs(df, scaler, FEAT_NAMES, results)
    else:
        st.markdown("""<div class="warn-box">
            <strong>advanced_module.py not found.</strong>
            Make sure <code>advanced_module.py</code> is in the same folder.
        </div>""", unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 8 — LIVE DIAGNOSIS
# └─────────────────────────────────────────────────────────────────────────────
with tabs[9]:
    st.markdown('<div class="section-title">Patient Biomarker Input</div>', unsafe_allow_html=True)

    PRESETS = {
        "Manual Entry":      {f: (FEATURE_META[f]["healthy"]+FEATURE_META[f]["pd"])/2 for f in FEAT_NAMES},
        "Healthy Reference": {f: FEATURE_META[f]["healthy"] for f in FEAT_NAMES},
        "PD Reference":      {f: FEATURE_META[f]["pd"]      for f in FEAT_NAMES},
        "Borderline Case":   {f: (FEATURE_META[f]["healthy"]+FEATURE_META[f]["pd"])/2 for f in FEAT_NAMES},
    }

    pcol1, pcol2 = st.columns([3,1])
    with pcol1: preset = st.radio("Load preset:", list(PRESETS.keys()), horizontal=True)
    with pcol2: diag_model = st.selectbox("Diagnosis model:", list(results.keys()), key="diag_model")
    init = PRESETS[preset]

    user_input = {}
    for cat in CATS:
        cat_feats = [f for f in FEAT_NAMES if FEATURE_META[f]["cat"]==cat]
        st.markdown(f"<div style='font-size:0.72rem;font-weight:700;color:{CAT_COLORS[cat]};letter-spacing:1px;margin:14px 0 8px 0;text-transform:uppercase;border-left:3px solid {CAT_COLORS[cat]};padding-left:8px'>{cat} Features</div>", unsafe_allow_html=True)
        cols = st.columns(min(len(cat_feats), 4))
        for col, f in zip(cols, cat_feats):
            m = FEATURE_META[f]
            with col:
                user_input[f] = st.slider(f, float(m["lo"]), float(m["hi"]),
                                           float(init.get(f,(m["lo"]+m["hi"])/2)),
                                           step=float((m["hi"]-m["lo"])/300),
                                           format=f"%{m['fmt']}", help=m["desc"],
                                           key=f"sl_{f}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬  Run Full Diagnostic", width="stretch"):
        X_in     = np.array([[user_input[f] for f in FEAT_NAMES]])
        X_scaled = scaler.transform(X_in)
        clf      = results[diag_model]["clf"]
        pred     = clf.predict(X_scaled)[0]
        proba    = clf.predict_proba(X_scaled)[0]

        votes         = {n: results[n]["clf"].predict(X_scaled)[0] for n in results}
        pd_votes      = sum(v==1 for v in votes.values())
        ensemble_prob = np.mean([results[n]["clf"].predict_proba(X_scaled)[0][1] for n in results])

        res_c1, res_c2, res_c3 = st.columns([1.2,1,1])
        with res_c1:
            if pred == 1:
                st.markdown(f"""<div class="result-pd">
                    <div style='font-size:3rem'>⚠️</div>
                    <div style='font-family:DM Serif Display,serif;font-size:1.8rem;color:{RED}'>Parkinson's Detected</div>
                    <div style='font-size:2.2rem;font-weight:700;color:{RED}'>{proba[1]*100:.1f}%</div>
                    <div style='color:{TEXT_DIM};font-size:0.72rem;letter-spacing:1px;margin-top:4px'>PROBABILITY — {diag_model.upper()}</div>
                    <div style='margin-top:10px;font-size:0.8rem;color:{TEXT_MID}'>Ensemble: <strong style='color:{RED}'>{ensemble_prob*100:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-healthy">
                    <div style='font-size:3rem'>✅</div>
                    <div style='font-family:DM Serif Display,serif;font-size:1.8rem;color:{GREEN}'>Healthy</div>
                    <div style='font-size:2.2rem;font-weight:700;color:{GREEN}'>{proba[0]*100:.1f}%</div>
                    <div style='color:{TEXT_DIM};font-size:0.72rem;letter-spacing:1px;margin-top:4px'>CONFIDENCE — {diag_model.upper()}</div>
                    <div style='margin-top:10px;font-size:0.8rem;color:{TEXT_MID}'>Ensemble: <strong style='color:{GREEN}'>{(1-ensemble_prob)*100:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)

        with res_c2:
            st.markdown('<div class="section-title">All Model Votes</div>', unsafe_allow_html=True)
            for name, vote in votes.items():
                icon = f'<span style="color:{RED};font-weight:600">⚠ PD</span>' if vote==1 else f'<span style="color:{GREEN};font-weight:600">✓ OK</span>'
                st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:5px 0;border-bottom:1px solid {BORDER}"><span style="color:{TEXT_MID}">{name}</span>{icon}</div>', unsafe_allow_html=True)
            st.markdown(f'<br><div style="font-size:0.9rem;font-weight:600">PD votes: <span style="color:{RED}">{pd_votes}/{len(results)}</span></div>', unsafe_allow_html=True)

        with res_c3:
            st.markdown('<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            ax.barh(["Healthy","PD"], [proba[0],proba[1]], color=[GREEN,RED], alpha=0.8, edgecolor=WHITE, height=0.5)
            for bar, v in zip(ax.patches, [proba[0],proba[1]]):
                ax.text(max(bar.get_width()-0.05,0.02), bar.get_y()+bar.get_height()/2,
                        f"{v*100:.1f}%", va="center", ha="right", fontsize=14, fontweight="bold",
                        color=WHITE if v>0.3 else TEXT_MAIN)
            ax.set(xlim=[0,1], title=diag_model); ax.grid(axis="x"); ax.set_xlabel("Probability")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Deviation chart
        st.markdown('<div class="section-title" style="margin-top:20px">Biomarker Deviation from Healthy Reference</div>', unsafe_allow_html=True)
        lo_arr = np.array([FEATURE_META[f]["lo"] for f in FEAT_NAMES])
        hi_arr = np.array([FEATURE_META[f]["hi"] for f in FEAT_NAMES])
        p_vals_arr = np.array([user_input[f] for f in FEAT_NAMES])
        h_vals_arr = np.array([FEATURE_META[f]["healthy"] for f in FEAT_NAMES])
        p_norm = (p_vals_arr-lo_arr)/(hi_arr-lo_arr)
        h_norm = (h_vals_arr-lo_arr)/(hi_arr-lo_arr)
        deviations = p_norm - h_norm
        bar_colors = [RED if d>0.05 else (GREEN if d<-0.05 else AMBER) for d in deviations]
        fig, ax = plt.subplots(figsize=(14,4))
        ax.bar(range(len(FEAT_NAMES)), deviations, color=bar_colors, alpha=0.8, edgecolor=WHITE, zorder=2)
        ax.axhline(0, color=NAVY, lw=1.5, zorder=3)
        ax.set_xticks(range(len(FEAT_NAMES))); ax.set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set(title="Normalized Deviation from Healthy Reference", ylabel="Δ (patient − healthy)")
        ax.grid(axis="y", zorder=0)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 7 — DATA EXPLORER
# └─────────────────────────────────────────────────────────────────────────────
with tabs[9]:
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div class="section-title">Raw Dataset</div>', unsafe_allow_html=True)
        disp = df.copy()
        disp.insert(0, "Status", disp.pop("status").map({0:"✅ Healthy",1:"⚠️ PD"}))
        st.dataframe(disp.head(40), width="stretch")
    with col2:
        st.markdown('<div class="section-title">Summary Stats</div>', unsafe_allow_html=True)
        st.dataframe(df[FEAT_NAMES].describe().round(4).T[["mean","std","min","max"]], width="stretch")

    st.markdown('<div class="section-title" style="margin-top:16px">Feature Pair Explorer</div>', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    feat_x = ca.selectbox("X-axis", FEAT_NAMES, index=0, key="fx")
    feat_y = cb.selectbox("Y-axis", FEAT_NAMES, index=7, key="fy")
    show_kde = cc.checkbox("Show KDE contours", value=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, name in [(0,GREEN,"Healthy"),(1,RED,"PD")]:
        mask = df.status==label
        ax.scatter(df[mask][feat_x], df[mask][feat_y], c=color, alpha=0.55, s=35,
                   edgecolors=WHITE, lw=0.4, label=name, zorder=3)
        if show_kde and mask.sum()>5:
            try:
                from scipy.stats import gaussian_kde
                xd,yd=df[mask][feat_x].values,df[mask][feat_y].values
                k=gaussian_kde(np.vstack([xd,yd]))
                xi,yi=np.mgrid[xd.min():xd.max():60j,yd.min():yd.max():60j]
                zi=k(np.vstack([xi.ravel(),yi.ravel()])).reshape(xi.shape)
                ax.contour(xi,yi,zi,levels=4,colors=[color],alpha=0.35,linewidths=1)
            except: pass
    ax.set(title=f"{feat_x}  vs  {feat_y}", xlabel=feat_x, ylabel=feat_y)
    ax.legend(); ax.grid(True)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
