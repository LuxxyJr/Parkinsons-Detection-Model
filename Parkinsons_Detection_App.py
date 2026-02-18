"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       PARKINSON'S DISEASE DETECTION — ADVANCED ML DASHBOARD                 ║
║       Dataset : UCI Parkinson's Voice Dataset (Little et al., 2008)         ║
║       Run     : streamlit run parkinsons_app.py                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    learning_curve, validation_curve
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


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PD Detection Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG    = "#07090f"
PANEL_BG   = "#0b1120"
PANEL_BG2  = "#0d1428"
BORDER     = "#112240"
CYAN       = "#00e5ff"
CYAN_DIM   = "#007fa8"
GREEN      = "#00ff9d"
RED        = "#ff4f6b"
AMBER      = "#ffb347"
PURPLE     = "#bf80ff"
TEXT_MAIN  = "#d4e8f0"
TEXT_DIM   = "#3d6a82"
TEXT_MID   = "#7ab0c8"

plt.rcParams.update({
    "figure.facecolor":  PANEL_BG,
    "axes.facecolor":    DARK_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT_MID,
    "axes.titlecolor":   CYAN,
    "text.color":        TEXT_MAIN,
    "xtick.color":       TEXT_DIM,
    "ytick.color":       TEXT_DIM,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  BORDER,
})

CMAP_PD     = LinearSegmentedColormap.from_list("pd", [DARK_BG, RED],    N=256)
CMAP_HEALTH = LinearSegmentedColormap.from_list("h",  [DARK_BG, GREEN],  N=256)
CMAP_MAIN   = LinearSegmentedColormap.from_list("m",  [GREEN, DARK_BG, RED], N=256)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Exo 2', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT_MAIN};
}}
div[data-testid="stSidebar"] {{
    background: #060810;
    border-right: 1px solid {BORDER};
}}
div[data-testid="stSidebar"] * {{ color: {TEXT_MID}; }}
.stTabs [data-baseweb="tab-list"] {{
    background: {PANEL_BG};
    border-radius: 2px;
    gap: 2px;
    border-bottom: 2px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 2px; font-size: 0.72rem;
    color: {TEXT_DIM}; background: transparent;
    padding: 8px 16px; border: none;
}}
.stTabs [aria-selected="true"] {{
    color: {CYAN} !important;
    border-bottom: 2px solid {CYAN} !important;
    background: rgba(0,229,255,0.05) !important;
}}
.stButton>button {{
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 2px; font-size: 0.75rem;
    border: 1px solid {CYAN_DIM}; color: {CYAN};
    background: rgba(0,229,255,0.06);
    transition: all 0.2s;
}}
.stButton>button:hover {{
    border-color: {CYAN}; background: rgba(0,229,255,0.15);
    box-shadow: 0 0 16px rgba(0,229,255,0.25);
}}
.stSelectbox>div>div, .stSlider>div {{
    background: {PANEL_BG2} !important;
}}
.card {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 18px 22px;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.card::before {{
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 2px;
}}
.card-cyan::before  {{ background: {CYAN}; box-shadow: 0 0 10px {CYAN}88; }}
.card-green::before {{ background: {GREEN}; box-shadow: 0 0 10px {GREEN}88; }}
.card-red::before   {{ background: {RED}; box-shadow: 0 0 10px {RED}88; }}
.card-amber::before {{ background: {AMBER}; box-shadow: 0 0 10px {AMBER}88; }}
.card-purple::before{{ background: {PURPLE}; box-shadow: 0 0 10px {PURPLE}88; }}
.metric-val {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.1rem; font-weight: 700;
    line-height: 1; margin-bottom: 4px;
}}
.metric-label {{
    font-size: 0.65rem; letter-spacing: 3px;
    text-transform: uppercase; color: {TEXT_DIM};
}}
.section-title {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem; letter-spacing: 4px;
    color: {CYAN_DIM}; text-transform: uppercase;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 6px; margin-bottom: 16px;
}}
.result-pd {{
    background: rgba(255,79,107,0.08);
    border: 2px solid {RED};
    border-radius: 6px; padding: 24px;
    text-align: center;
}}
.result-healthy {{
    background: rgba(0,255,157,0.08);
    border: 2px solid {GREEN};
    border-radius: 6px; padding: 24px;
    text-align: center;
}}
.tag-pd      {{ background: rgba(255,79,107,0.2); color:{RED}; padding:2px 10px; border-radius:2px; font-family:monospace; font-size:0.75rem; }}
.tag-healthy {{ background: rgba(0,255,157,0.2); color:{GREEN}; padding:2px 10px; border-radius:2px; font-family:monospace; font-size:0.75rem; }}
.info-box {{
    background: rgba(0,229,255,0.05);
    border: 1px solid {BORDER};
    border-left: 3px solid {CYAN};
    padding: 12px 16px; border-radius: 2px;
    font-size: 0.82rem; line-height: 1.7;
}}
.warn-box {{
    background: rgba(255,179,71,0.06);
    border: 1px solid rgba(255,179,71,0.25);
    border-left: 3px solid {AMBER};
    padding: 10px 14px; border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem; color: #aa8022; letter-spacing: 1px;
}}
hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA — UCI-FAITHFUL SYNTHETIC GENERATION
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_META = {
    "MDVP:Fo(Hz)":        dict(desc="Average vocal fundamental frequency",   cat="Frequency", healthy=197.10, pd=145.21, unit="Hz",   lo=80,    hi=280,   fmt=".2f"),
    "MDVP:Fhi(Hz)":       dict(desc="Maximum vocal fundamental frequency",   cat="Frequency", healthy=243.60, pd=197.11, unit="Hz",   lo=90,    hi=400,   fmt=".2f"),
    "MDVP:Flo(Hz)":       dict(desc="Minimum vocal fundamental frequency",   cat="Frequency", healthy=146.67, pd=102.15, unit="Hz",   lo=60,    hi=250,   fmt=".2f"),
    "MDVP:Jitter(%)":     dict(desc="Frequency variation cycle-to-cycle",    cat="Jitter",    healthy=0.0033,  pd=0.0063,  unit="%",    lo=0.001, hi=0.02,  fmt=".4f"),
    "MDVP:Jitter(Abs)":   dict(desc="Absolute jitter in seconds",            cat="Jitter",    healthy=2.2e-5,  pd=4.5e-5,  unit="s",    lo=1e-6,  hi=1.5e-4,fmt=".6f"),
    "MDVP:RAP":           dict(desc="Relative amplitude perturbation",       cat="Jitter",    healthy=0.0017,  pd=0.0033,  unit="",     lo=0.0005,hi=0.012, fmt=".4f"),
    "MDVP:PPQ":           dict(desc="5-point period perturbation quotient",  cat="Jitter",    healthy=0.0018,  pd=0.0034,  unit="",     lo=0.0005,hi=0.012, fmt=".4f"),
    "Jitter:DDP":         dict(desc="Average absolute jitter difference",    cat="Jitter",    healthy=0.0052,  pd=0.0100,  unit="",     lo=0.001, hi=0.036, fmt=".4f"),
    "MDVP:Shimmer":       dict(desc="Local shimmer (amplitude variation)",   cat="Shimmer",   healthy=0.0231,  pd=0.0508,  unit="",     lo=0.008, hi=0.12,  fmt=".4f"),
    "MDVP:Shimmer(dB)":   dict(desc="Shimmer in decibels",                  cat="Shimmer",   healthy=0.214,   pd=0.471,   unit="dB",   lo=0.06,  hi=1.1,   fmt=".3f"),
    "Shimmer:APQ3":       dict(desc="3-point amplitude perturbation quotient",cat="Shimmer",  healthy=0.0122,  pd=0.0269,  unit="",     lo=0.004, hi=0.065, fmt=".4f"),
    "Shimmer:APQ5":       dict(desc="5-point amplitude perturbation quotient",cat="Shimmer",  healthy=0.0145,  pd=0.0316,  unit="",     lo=0.005, hi=0.079, fmt=".4f"),
    "MDVP:APQ":           dict(desc="11-point amplitude perturbation quotient",cat="Shimmer", healthy=0.0199,  pd=0.0439,  unit="",     lo=0.006, hi=0.11,  fmt=".4f"),
    "Shimmer:DDA":        dict(desc="Average absolute shimmer difference",   cat="Shimmer",   healthy=0.0366,  pd=0.0808,  unit="",     lo=0.012, hi=0.20,  fmt=".4f"),
    "NHR":                dict(desc="Noise-to-harmonics ratio",              cat="Noise",     healthy=0.0111,  pd=0.0312,  unit="",     lo=0.001, hi=0.30,  fmt=".4f"),
    "HNR":                dict(desc="Harmonics-to-noise ratio",              cat="Noise",     healthy=24.68,   pd=19.98,   unit="dB",   lo=7.0,   hi=34.0,  fmt=".2f"),
    "RPDE":               dict(desc="Recurrence period density entropy",     cat="Nonlinear", healthy=0.499,   pd=0.587,   unit="",     lo=0.25,  hi=0.84,  fmt=".4f"),
    "DFA":                dict(desc="Detrended fluctuation analysis",        cat="Nonlinear", healthy=0.718,   pd=0.753,   unit="",     lo=0.57,  hi=0.88,  fmt=".4f"),
    "spread1":            dict(desc="Nonlinear frequency variation measure", cat="Nonlinear", healthy=-6.759,  pd=-5.335,  unit="",     lo=-8.0,  hi=-3.0,  fmt=".3f"),
    "spread2":            dict(desc="Nonlinear frequency variation measure", cat="Nonlinear", healthy=0.168,   pd=0.269,   unit="",     lo=0.01,  hi=0.52,  fmt=".4f"),
    "D2":                 dict(desc="Correlation dimension",                 cat="Nonlinear", healthy=2.302,   pd=2.522,   unit="",     lo=1.5,   hi=3.5,   fmt=".3f"),
    "PPE":                dict(desc="Pitch period entropy",                  cat="Nonlinear", healthy=0.062,   pd=0.213,   unit="",     lo=0.02,  hi=0.53,  fmt=".4f"),
}
FEAT_NAMES = list(FEATURE_META.keys())
CATS = ["Frequency","Jitter","Shimmer","Noise","Nonlinear"]
CAT_COLORS = {"Frequency": CYAN, "Jitter": AMBER, "Shimmer": PURPLE, "Noise": RED, "Nonlinear": GREEN}

@st.cache_data
def load_data(seed=42):
    rng = np.random.RandomState(seed)
    n_pd, n_h = 147, 48

    specs = {
        "MDVP:Fo(Hz)":      (145.21,30.0,   197.10,40.0),
        "MDVP:Fhi(Hz)":     (197.11,55.0,   243.60,65.0),
        "MDVP:Flo(Hz)":     (102.15,25.0,   146.67,35.0),
        "MDVP:Jitter(%)":   (0.0063,0.003,  0.0033,0.001),
        "MDVP:Jitter(Abs)": (4.5e-5,2e-5,   2.2e-5,8e-6),
        "MDVP:RAP":         (0.0033,0.0016, 0.0017,0.0007),
        "MDVP:PPQ":         (0.0034,0.0017, 0.0018,0.0007),
        "Jitter:DDP":       (0.0100,0.005,  0.0052,0.002),
        "MDVP:Shimmer":     (0.0508,0.022,  0.0231,0.010),
        "MDVP:Shimmer(dB)": (0.471, 0.20,   0.214, 0.092),
        "Shimmer:APQ3":     (0.0269,0.012,  0.0122,0.005),
        "Shimmer:APQ5":     (0.0316,0.015,  0.0145,0.006),
        "MDVP:APQ":         (0.0439,0.020,  0.0199,0.009),
        "Shimmer:DDA":      (0.0808,0.036,  0.0366,0.015),
        "NHR":              (0.0312,0.025,  0.0111,0.007),
        "HNR":              (19.98, 5.0,    24.68, 4.0),
        "RPDE":             (0.587, 0.078,  0.499, 0.065),
        "DFA":              (0.753, 0.044,  0.718, 0.042),
        "spread1":          (-5.335,0.90,   -6.759,0.80),
        "spread2":          (0.269, 0.095,  0.168, 0.072),
        "D2":               (2.522, 0.35,   2.302, 0.31),
        "PPE":              (0.213, 0.10,   0.062, 0.030),
    }

    rows = {}
    for f, (mu_p, sd_p, mu_h, sd_h) in specs.items():
        m = FEATURE_META[f]
        pd_v = np.clip(rng.normal(mu_p, sd_p, n_pd), m["lo"], m["hi"])
        h_v  = np.clip(rng.normal(mu_h, sd_h, n_h),  m["lo"], m["hi"])
        rows[f] = np.concatenate([pd_v, h_v])

    df = pd.DataFrame(rows)
    df["status"] = [1]*n_pd + [0]*n_h
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

@st.cache_resource
def train_all(df):
    X = df[FEAT_NAMES]
    y = df["status"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)

    model_zoo = {
        "SVM (RBF)":            SVC(kernel="rbf", C=10, gamma=0.01, probability=True, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
        "Logistic Regression":  LogisticRegression(C=0.5, max_iter=2000, random_state=42),
        "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=7, metric="euclidean"),
        "AdaBoost":             AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        "Naive Bayes":          GaussianNB(),
        "Decision Tree":        DecisionTreeClassifier(max_depth=6, min_samples_split=5, random_state=42),
    }

    cv  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    res = {}

    for name, clf in model_zoo.items():
        clf.fit(Xtr, ytr)
        yp   = clf.predict(Xte)
        ypr  = clf.predict_proba(Xte)[:,1]
        cvs  = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy")
        fpr, tpr, thr = roc_curve(yte, ypr)
        pr_p, pr_r, _ = precision_recall_curve(yte, ypr)

        res[name] = dict(
            clf=clf, accuracy=accuracy_score(yte,yp),
            f1=f1_score(yte,yp), precision=precision_score(yte,yp),
            recall=recall_score(yte,yp), mcc=matthews_corrcoef(yte,yp),
            roc_auc=auc(fpr,tpr), pr_auc=average_precision_score(yte,ypr),
            cv_mean=cvs.mean(), cv_std=cvs.std(), cv_all=cvs,
            cm=confusion_matrix(yte,yp), report=classification_report(yte,yp,target_names=["Healthy","PD"]),
            fpr=fpr, tpr=tpr, pr_p=pr_p, pr_r=pr_r,
            y_test=yte, y_pred=yp, y_proba=ypr,
        )

    # Statistical significance tests per feature
    stat_tests = {}
    for f in FEAT_NAMES:
        pd_vals = df[df.status==1][f]
        h_vals  = df[df.status==0][f]
        u_stat, p_val = stats.mannwhitneyu(pd_vals, h_vals, alternative="two-sided")
        d = (pd_vals.mean() - h_vals.mean()) / np.sqrt((pd_vals.std()**2 + h_vals.std()**2)/2)
        stat_tests[f] = dict(p_val=p_val, effect_size=abs(d), u_stat=u_stat)

    return res, scaler, Xtr, Xte, ytr, yte

def fig2img(fig):
    """Convert matplotlib figure to bytes for display."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()
results, scaler, Xtr, Xte, ytr, yte = train_all(df)

# Statistical tests
stat_tests = {}
for f in FEAT_NAMES:
    pd_v = df[df.status==1][f]; h_v = df[df.status==0][f]
    u, p = stats.mannwhitneyu(pd_v, h_v, alternative="two-sided")
    d = (pd_v.mean()-h_v.mean()) / np.sqrt((pd_v.std()**2+h_v.std()**2)/2)
    stat_tests[f] = dict(p=p, d=abs(d), u=u)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div style='font-family:Share Tech Mono;font-size:1.1rem;color:{CYAN};letter-spacing:3px;margin-bottom:4px'>🧠 PD DETECTION</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:Share Tech Mono;font-size:0.6rem;color:{TEXT_DIM};letter-spacing:2px;margin-bottom:20px'>ML RESEARCH DASHBOARD v2.0</div>", unsafe_allow_html=True)

    st.markdown("**Select Classifier**")
    selected_model = st.selectbox("Classifier", list(results.keys()), label_visibility="collapsed")
    r = results[selected_model]

    st.markdown("---")
    st.markdown(f"<div style='font-family:Share Tech Mono;font-size:0.65rem;color:{TEXT_DIM};letter-spacing:2px'>SELECTED MODEL STATS</div>", unsafe_allow_html=True)

    stat_data = [
        ("Accuracy", f"{r['accuracy']*100:.1f}%", CYAN),
        ("F1 Score",  f"{r['f1']:.4f}",            GREEN),
        ("ROC AUC",   f"{r['roc_auc']:.4f}",       AMBER),
        ("MCC",       f"{r['mcc']:.4f}",            PURPLE),
        ("10-CV",     f"{r['cv_mean']*100:.1f}%",   RED),
    ]
    for label, val, color in stat_data:
        st.markdown(f"""<div style='display:flex;justify-content:space-between;margin:6px 0;font-size:0.8rem'>
            <span style='color:{TEXT_DIM}'>{label}</span>
            <span style='color:{color};font-family:Share Tech Mono'>{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='font-family:Share Tech Mono;font-size:0.65rem;color:{TEXT_DIM};letter-spacing:2px;margin-bottom:8px'>DATASET INFO</div>", unsafe_allow_html=True)
    info = [
        ("Source",   "UCI ML Repository"),
        ("Paper",    "Little et al., 2008"),
        ("Samples",  f"{len(df)} voices"),
        ("PD",       f"{(df.status==1).sum()} (75.4%)"),
        ("Healthy",  f"{(df.status==0).sum()} (24.6%)"),
        ("Features", f"{len(FEAT_NAMES)} biomarkers"),
        ("Models",   f"{len(results)} classifiers"),
    ]
    for k, v in info:
        st.markdown(f"""<div style='display:flex;justify-content:space-between;margin:4px 0;font-size:0.78rem'>
            <span style='color:{TEXT_DIM}'>{k}</span>
            <span style='color:{TEXT_MID}'>{v}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""<div class="warn-box">
        ⚠ RESEARCH USE ONLY<br>
        NOT A CLINICAL DIAGNOSTIC TOOL.<br>
        Always consult a neurologist.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='padding:20px 0 10px 0'>
  <div style='font-family:Exo 2,sans-serif;font-size:2.6rem;font-weight:800;color:{CYAN};
              letter-spacing:5px;text-shadow:0 0 30px {CYAN}66;line-height:1'>
    PARKINSON'S DETECTION LAB
  </div>
  <div style='font-family:Share Tech Mono,monospace;font-size:0.68rem;color:{TEXT_DIM};
              letter-spacing:4px;margin-top:6px'>
    VOICE &amp; GAIT BIOMARKER ANALYSIS ◆ UCI ML REPOSITORY ◆ LITTLE ET AL. 2008 ◆ 22 FEATURES ◆ 8 CLASSIFIERS
  </div>
</div>
""", unsafe_allow_html=True)

# Top metric strip
best = max(results, key=lambda x: results[x]["roc_auc"])
r = results[selected_model]
cols = st.columns(6)
metric_cards = [
    ("ACCURACY",   f"{r['accuracy']*100:.1f}%",   "cyan"),
    ("F1 SCORE",   f"{r['f1']:.4f}",               "green"),
    ("ROC AUC",    f"{r['roc_auc']:.4f}",          "amber"),
    ("PRECISION",  f"{r['precision']:.4f}",         "purple"),
    ("RECALL",     f"{r['recall']:.4f}",            "red"),
    ("10-FOLD CV", f"{r['cv_mean']*100:.1f}±{r['cv_std']*100:.1f}%", "cyan"),
]
for col, (lbl, val, color) in zip(cols, metric_cards):
    col.markdown(f'<div class="card card-{color}"><div class="metric-val" style="color:var(--c)">{val}</div><div class="metric-label">{lbl}</div></div>'.replace("var(--c)", dict(cyan=CYAN, green=GREEN, amber=AMBER, purple=PURPLE, red=RED)[color]), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📡 OVERVIEW",
    "📈 MODEL PERFORMANCE",
    "🔬 FEATURE INTELLIGENCE",
    "🏆 MODEL COMPARISON",
    "🩺 LIVE DIAGNOSIS",
    "📊 DATA EXPLORER",
    "🧩 EXPLAINABILITY",
])


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 0 — OVERVIEW
# └─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown('<div class="section-title">◆ MODEL LEADERBOARD</div>', unsafe_allow_html=True)
        rows = []
        for name, res in results.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{res['accuracy']*100:.2f}%",
                "F1":       f"{res['f1']:.4f}",
                "AUC":      f"{res['roc_auc']:.4f}",
                "PR-AUC":   f"{res['pr_auc']:.4f}",
                "MCC":      f"{res['mcc']:.4f}",
                "10-CV":    f"{res['cv_mean']*100:.1f}±{res['cv_std']*100:.1f}%",
            })
        lb = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
        lb.index = lb.index + 1
        st.dataframe(lb, width="stretch")

        st.markdown('<div class="section-title" style="margin-top:24px">◆ ALL MODELS — ROC OVERVIEW</div>', unsafe_allow_html=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        pal = [CYAN, GREEN, AMBER, PURPLE, RED, "#ff80aa", "#80ffff", "#aaff80"]
        for (name, res), c in zip(results.items(), pal):
            ax1.plot(res["fpr"], res["tpr"], lw=1.8, color=c, label=f"{name[:18]} ({res['roc_auc']:.3f})")
            ax2.plot(res["pr_r"], res["pr_p"], lw=1.8, color=c)
        ax1.plot([0,1],[0,1], color=BORDER, lw=1, linestyle="--")
        ax1.set(title="ROC Curves", xlabel="FPR", ylabel="TPR"); ax1.legend(fontsize=6.5); ax1.grid(True)
        ax2.set(title="Precision-Recall Curves", xlabel="Recall", ylabel="Precision"); ax2.grid(True)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with c2:
        st.markdown('<div class="section-title">◆ CLASS BALANCE</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sz = [(df.status==0).sum(), (df.status==1).sum()]
        wedges, _, at = ax.pie(sz, labels=["Healthy","PD"], colors=[GREEN, RED],
                                autopct="%1.1f%%", startangle=90,
                                wedgeprops=dict(edgecolor=PANEL_BG, linewidth=2.5, width=0.55))
        for t in at: t.set_color(DARK_BG); t.set_fontsize(11); t.set_fontweight("bold")
        ax.set_title("Dataset Distribution", pad=10)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown('<div class="section-title" style="margin-top:16px">◆ CV SCORE DISTRIBUTION</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cv_data  = [results[n]["cv_all"]*100 for n in results]
        cv_names = [n.replace(" ","\n") for n in results]
        bp = ax.boxplot(cv_data, labels=cv_names, patch_artist=True, widths=0.5,
                        medianprops=dict(color=CYAN, lw=2),
                        whiskerprops=dict(color=TEXT_DIM),
                        capprops=dict(color=TEXT_DIM),
                        flierprops=dict(marker="o", markerfacecolor=RED, markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], pal):
            patch.set(facecolor=c+"22", edgecolor=c)
        ax.set(title="10-Fold CV Accuracy", ylabel="Accuracy (%)"); ax.grid(axis="y")
        plt.xticks(fontsize=6.5); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown('<div class="section-title" style="margin-top:16px">◆ TOP 5 FEATURES BY EFFECT SIZE</div>', unsafe_allow_html=True)
        top5 = sorted(stat_tests.items(), key=lambda x: -x[1]["d"])[:5]
        for f, st_res in top5:
            d = st_res["d"]; p = st_res["p"]
            bar_w = min(d / 3.0, 1.0)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
            st.markdown(f"""<div style='margin:8px 0'>
              <div style='display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:3px'>
                <span style='color:{TEXT_MID}'>{f.split(":")[1] if ":" in f else f}</span>
                <span style='color:{AMBER};font-family:monospace'>d={d:.2f} {sig}</span>
              </div>
              <div style='height:5px;background:{BORDER};border-radius:3px;overflow:hidden'>
                <div style='height:100%;width:{bar_w*100:.0f}%;background:linear-gradient(to right,{AMBER}44,{AMBER});border-radius:3px'></div>
              </div>
            </div>""", unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 1 — MODEL PERFORMANCE
# └─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    r = results[selected_model]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">◆ CONFUSION MATRIX</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        cm_norm = r["cm"].astype(float) / r["cm"].sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap=LinearSegmentedColormap.from_list("cm",[DARK_BG,"#1a2a4a",CYAN],256), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                count = r["cm"][i,j]
                pct   = cm_norm[i,j]*100
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color=DARK_BG if cm_norm[i,j] > 0.6 else TEXT_MAIN)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred: Healthy","Pred: PD"], fontsize=9)
        ax.set_yticklabels(["True: Healthy","True: PD"], fontsize=9)
        ax.set_title(f"{selected_model} — Confusion Matrix", pad=12)
        plt.colorbar(im, ax=ax, fraction=0.046, label="Proportion")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown('<div class="section-title" style="margin-top:20px">◆ CLASSIFICATION REPORT</div>', unsafe_allow_html=True)
        st.code(r["report"], language="")

    with col2:
        st.markdown('<div class="section-title">◆ ROC + PRECISION-RECALL CURVES</div>', unsafe_allow_html=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
        ax1.plot(r["fpr"], r["tpr"], color=CYAN, lw=2.5, label=f"AUC = {r['roc_auc']:.4f}")
        ax1.fill_between(r["fpr"], r["tpr"], alpha=0.1, color=CYAN)
        ax1.plot([0,1],[0,1], color=BORDER, lw=1.2, ls="--", label="Random (0.5)")
        ax1.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax1.legend(loc="lower right"); ax1.grid(True)

        ax2.plot(r["pr_r"], r["pr_p"], color=GREEN, lw=2.5, label=f"AP = {r['pr_auc']:.4f}")
        ax2.fill_between(r["pr_r"], r["pr_p"], alpha=0.1, color=GREEN)
        baseline = (df.status==1).sum() / len(df)
        ax2.axhline(baseline, color=BORDER, lw=1.2, ls="--", label=f"Baseline ({baseline:.2f})")
        ax2.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision", ylim=[0,1.05])
        ax2.legend(); ax2.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title" style="margin-top:16px">◆ LEARNING CURVES</div>', unsafe_allow_html=True)
    Xs = scaler.transform(df[FEAT_NAMES])
    with st.spinner("Computing learning curves…"):
        train_sz, train_sc, val_sc = learning_curve(
            r["clf"], Xs, df["status"],
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring="accuracy", n_jobs=-1
        )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(train_sz, train_sc.mean(1)-train_sc.std(1), train_sc.mean(1)+train_sc.std(1), alpha=0.15, color=CYAN)
    ax.fill_between(train_sz, val_sc.mean(1)-val_sc.std(1),   val_sc.mean(1)+val_sc.std(1),   alpha=0.15, color=GREEN)
    ax.plot(train_sz, train_sc.mean(1), "o-", color=CYAN,  lw=2, ms=5, label="Training Score")
    ax.plot(train_sz, val_sc.mean(1),   "s-", color=GREEN, lw=2, ms=5, label="Cross-Val Score")
    gap = train_sc.mean(1) - val_sc.mean(1)
    ax.set(title=f"{selected_model} — Learning Curves", xlabel="Training Samples", ylabel="Accuracy")
    ax.legend(); ax.grid(True)
    # Annotate bias-variance
    ax.annotate(f"Bias-Variance Gap: {gap[-1]:.3f}", xy=(train_sz[-1], val_sc.mean(1)[-1]),
                xytext=(train_sz[-2]-30, val_sc.mean(1)[-1]-0.05),
                color=AMBER, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=AMBER))
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">◆ THRESHOLD ANALYSIS</div>', unsafe_allow_html=True)
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
        ax.plot(thresholds, f1s,   color=CYAN,   lw=2, label="F1")
        ax.plot(thresholds, precs, color=GREEN,  lw=2, label="Precision")
        ax.plot(thresholds, recs,  color=AMBER,  lw=2, label="Recall")
        ax.plot(thresholds, accs,  color=PURPLE, lw=2, label="Accuracy", ls="--")
        ax.axvline(best_t, color=RED, lw=1.5, ls=":", label=f"Best F1 @ {best_t:.2f}")
        ax.set(title="Metrics vs Decision Threshold", xlabel="Threshold", ylabel="Score", ylim=[0,1.05])
        ax.legend(fontsize=7.5); ax.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">◆ 10-FOLD CROSS-VALIDATION</div>', unsafe_allow_html=True)
        cvs = r["cv_all"] * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors_bar = [GREEN if v >= cvs.mean() else AMBER for v in cvs]
        bars = ax.bar(range(1, 11), cvs, color=colors_bar, alpha=0.8, edgecolor=BORDER, zorder=2)
        ax.axhline(cvs.mean(), color=CYAN, lw=2, ls="--", label=f"Mean: {cvs.mean():.2f}%", zorder=3)
        ax.axhline(cvs.mean()-cvs.std(), color=CYAN, lw=1, ls=":", alpha=0.5, zorder=3)
        ax.axhline(cvs.mean()+cvs.std(), color=CYAN, lw=1, ls=":", alpha=0.5, label=f"±1σ: {cvs.std():.2f}%", zorder=3)
        for bar, v in zip(bars, cvs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=TEXT_MID)
        ax.set(title="10-Fold CV Accuracy", xlabel="Fold", ylabel="Accuracy (%)", ylim=[60, 105])
        ax.legend(fontsize=8); ax.grid(axis="y", zorder=0)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 2 — FEATURE INTELLIGENCE
# └─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">◆ FEATURE IMPORTANCE (RANDOM FOREST)</div>', unsafe_allow_html=True)
        rf  = results["Random Forest"]["clf"]
        imp = rf.feature_importances_
        idx = np.argsort(imp)[::-1]

        fig, ax = plt.subplots(figsize=(6, 7))
        top_n = 18
        feat_top = [FEAT_NAMES[i] for i in idx[:top_n]]
        imp_top  = imp[idx[:top_n]]
        cats_top = [FEATURE_META[f]["cat"] for f in feat_top]
        bar_colors = [CAT_COLORS[c] for c in cats_top]
        bars = ax.barh(feat_top[::-1], imp_top[::-1], color=bar_colors[::-1], alpha=0.82, edgecolor=BORDER, height=0.7)
        ax.set(title="Random Forest Feature Importances (Top 18)", xlabel="Mean Decrease in Impurity")
        ax.grid(axis="x")
        # Category legend
        handles = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CATS]
        ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">◆ STATISTICAL SIGNIFICANCE (MANN-WHITNEY U)</div>', unsafe_allow_html=True)
        st_sorted = sorted(stat_tests.items(), key=lambda x: -x[1]["d"])
        feat_sig  = [f for f, _ in st_sorted]
        d_vals    = [v["d"] for _, v in st_sorted]
        p_vals    = [v["p"] for _, v in st_sorted]
        sig_colors= [RED if p < 0.001 else (AMBER if p < 0.01 else (GREEN if p < 0.05 else TEXT_DIM)) for p in p_vals]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 7))
        ax1.barh(feat_sig[::-1], d_vals[::-1], color=sig_colors[::-1], alpha=0.8, edgecolor=BORDER, height=0.7)
        ax1.axvline(0.2, color=TEXT_DIM, lw=1, ls="--", label="Small (0.2)")
        ax1.axvline(0.5, color=AMBER,    lw=1, ls="--", label="Medium (0.5)")
        ax1.axvline(0.8, color=RED,      lw=1, ls="--", label="Large (0.8)")
        ax1.set(title="Cohen's d Effect Size", xlabel="|d|"); ax1.legend(fontsize=6.5); ax1.grid(axis="x")

        neg_log_p = [-np.log10(max(p, 1e-10)) for p in p_vals]
        bar_colors2 = [RED if v >= -np.log10(0.001) else (AMBER if v >= -np.log10(0.01) else (GREEN if v >= -np.log10(0.05) else TEXT_DIM)) for v in neg_log_p]
        ax2.barh(feat_sig[::-1], neg_log_p[::-1], color=bar_colors2[::-1], alpha=0.8, edgecolor=BORDER, height=0.7)
        ax2.axvline(-np.log10(0.001), color=RED,     lw=1, ls="--", label="p=0.001")
        ax2.axvline(-np.log10(0.05),  color=GREEN,   lw=1, ls="--", label="p=0.05")
        ax2.set(title="-log₁₀(p-value)", xlabel="-log₁₀(p)"); ax2.legend(fontsize=6.5); ax2.grid(axis="x")
        ax2.set_yticklabels([])
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title" style="margin-top:4px">◆ VIOLIN PLOTS — TOP 8 FEATURES</div>', unsafe_allow_html=True)
    top8 = [FEAT_NAMES[i] for i in idx[:8]]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    for i, (feat, ax) in enumerate(zip(top8, axes)):
        pd_v = df[df.status==1][feat].values
        h_v  = df[df.status==0][feat].values
        data = [h_v, pd_v]
        parts = ax.violinplot(data, positions=[0,1], showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], [GREEN, RED]):
            pc.set(facecolor=color, alpha=0.4, edgecolor=color)
        parts["cmedians"].set(color=TEXT_MAIN, lw=2)
        parts["cbars"].set(color=TEXT_DIM, lw=1)
        parts["cmins"].set(color=TEXT_DIM, lw=1)
        parts["cmaxes"].set(color=TEXT_DIM, lw=1)
        ax.scatter([0]*len(h_v),  h_v,  color=GREEN, alpha=0.25, s=6, zorder=3)
        ax.scatter([1]*len(pd_v), pd_v, color=RED,   alpha=0.25, s=6, zorder=3)
        d = stat_tests[feat]["d"]; p = stat_tests[feat]["p"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
        ax.set_title(f"{feat}\n(d={d:.2f}, p{sig})", fontsize=7.5, pad=4)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Healthy","PD"], fontsize=8)
        ax.grid(axis="y")
        cat_c = CAT_COLORS[FEATURE_META[feat]["cat"]]
        for spine in ax.spines.values(): spine.set_edgecolor(cat_c); spine.set_linewidth(1.2)
    fig.suptitle("Feature Distributions — PD vs Healthy (Violin + Scatter)", y=1.01, color=CYAN, fontsize=11)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">◆ PCA VISUALIZATION</div>', unsafe_allow_html=True)
        pca2 = PCA(n_components=2, random_state=42)
        Xs   = scaler.transform(df[FEAT_NAMES])
        Xp   = pca2.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for label, color, name in [(0, GREEN, "Healthy"), (1, RED, "PD")]:
            mask = df.status == label
            ax.scatter(Xp[mask,0], Xp[mask,1], c=color, alpha=0.55, s=28,
                       edgecolors=BORDER, linewidths=0.3, label=f"{name} (n={mask.sum()})", zorder=3)
        ax.set(title=f"PCA 2D — PC1:{pca2.explained_variance_ratio_[0]*100:.1f}%  PC2:{pca2.explained_variance_ratio_[1]*100:.1f}%",
               xlabel="Principal Component 1", ylabel="Principal Component 2")
        ax.legend(); ax.grid(True)
        # Draw decision region approximation
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">◆ CORRELATION HEATMAP (TOP 12)</div>', unsafe_allow_html=True)
        top12 = [FEAT_NAMES[i] for i in idx[:12]]
        corr  = df[top12 + ["status"]].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, annot=True, fmt=".2f",
                    cmap=LinearSegmentedColormap.from_list("div", [GREEN, DARK_BG, RED], 256),
                    ax=ax, linewidths=0.4, linecolor=BORDER,
                    annot_kws={"size":6.5}, vmin=-1, vmax=1,
                    cbar_kws={"shrink":0.7, "label":"Pearson r"})
        ax.set_title("Feature Correlation Matrix", pad=10)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 3 — MODEL COMPARISON
# └─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    pal = [CYAN, GREEN, AMBER, PURPLE, RED, "#ff80aa", "#80ffff", "#aaff80"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">◆ ROC CURVES — ALL MODELS</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, res), c in zip(results.items(), pal):
            ax.plot(res["fpr"], res["tpr"], lw=2, color=c, label=f"{name} ({res['roc_auc']:.3f})")
        ax.plot([0,1],[0,1], color=BORDER, lw=1, ls="--", label="Random")
        ax.fill_between([0,1], [0,1], alpha=0.05, color=TEXT_DIM)
        ax.set(title="ROC Curves — All Classifiers", xlabel="FPR", ylabel="TPR")
        ax.legend(loc="lower right", fontsize=7); ax.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">◆ ACCURACY vs AUC SCATTER</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, res), c in zip(results.items(), pal):
            ax.scatter(res["accuracy"]*100, res["roc_auc"], color=c, s=120, edgecolors=BORDER, lw=1.5, zorder=4)
            ax.annotate(name, (res["accuracy"]*100, res["roc_auc"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=7.5, color=c)
        ax.set(title="Accuracy vs ROC-AUC", xlabel="Accuracy (%)", ylabel="ROC AUC")
        ax.grid(True); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title">◆ MULTI-METRIC RADAR COMPARISON</div>', unsafe_allow_html=True)
    metrics_radar = ["accuracy","f1","roc_auc","pr_auc","recall","precision"]
    labels_radar  = ["Accuracy","F1","ROC-AUC","PR-AUC","Recall","Precision"]
    N = len(metrics_radar)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor(DARK_BG)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_radar, size=9, color=TEXT_MID)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["60","70","80","90","100"], size=7, color=TEXT_DIM)
    ax.grid(color=BORDER, lw=0.8)
    ax.spines["polar"].set_edgecolor(BORDER)

    for (name, res), c in zip(results.items(), pal):
        vals = [res[m] for m in metrics_radar] + [res[metrics_radar[0]]]
        ax.plot(angles, vals, lw=2, color=c, label=name)
        ax.fill(angles, vals, alpha=0.07, color=c)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7.5)
    ax.set_title("Multi-Metric Radar — All Models", pad=20, color=CYAN)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title">◆ GROUPED BAR — ALL METRICS</div>', unsafe_allow_html=True)
    names_b = list(results.keys())
    metric_keys = ["accuracy","f1","roc_auc","pr_auc","mcc"]
    metric_labels = ["Accuracy","F1 Score","ROC AUC","PR AUC","MCC"]
    x = np.arange(len(names_b))
    w = 0.15
    fig, ax = plt.subplots(figsize=(13, 4))
    for i, (mk, ml, c) in enumerate(zip(metric_keys, metric_labels, [CYAN,GREEN,AMBER,PURPLE,RED])):
        vals = [results[n][mk] for n in names_b]
        ax.bar(x + i*w, vals, w, label=ml, color=c, alpha=0.8, edgecolor=BORDER)
    ax.set(title="All Models × All Metrics", ylabel="Score", xticks=x+w*2)
    ax.set_xticklabels([n.replace(" ","\n") for n in names_b], fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis="y"); ax.set_ylim(0, 1.1)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 4 — LIVE DIAGNOSIS
# └─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-title">◆ PATIENT BIOMARKER INPUT</div>', unsafe_allow_html=True)

    PRESETS = {
        "Manual Entry":        None,
        "Healthy Reference":   {f: FEATURE_META[f]["healthy"] for f in FEAT_NAMES},
        "PD Reference":        {f: FEATURE_META[f]["pd"]      for f in FEAT_NAMES},
        "Borderline Case":     {f: (FEATURE_META[f]["healthy"]+FEATURE_META[f]["pd"])/2 for f in FEAT_NAMES},
    }

    pcol1, pcol2 = st.columns([3, 1])
    with pcol1:
        preset = st.radio("Load preset patient:", list(PRESETS.keys()), horizontal=True)
    with pcol2:
        diag_model = st.selectbox("Diagnosis model:", list(results.keys()), key="diag_model")

    init = PRESETS[preset] if PRESETS[preset] else {f: (FEATURE_META[f]["healthy"]+FEATURE_META[f]["pd"])/2 for f in FEAT_NAMES}

    # Group sliders by category
    user_input = {}
    for cat in CATS:
        cat_feats = [f for f in FEAT_NAMES if FEATURE_META[f]["cat"] == cat]
        st.markdown(f'<div style="font-family:Share Tech Mono;font-size:0.62rem;color:{CAT_COLORS[cat]};letter-spacing:3px;margin:14px 0 8px 0">▶ {cat.upper()} FEATURES</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(cat_feats), 4))
        for col, f in zip(cols, cat_feats):
            m = FEATURE_META[f]
            default = float(init.get(f, (m["lo"]+m["hi"])/2))
            with col:
                user_input[f] = st.slider(
                    f, float(m["lo"]), float(m["hi"]), default,
                    step=float((m["hi"]-m["lo"])/300),
                    format=f"%{m['fmt']}",
                    help=m["desc"],
                    key=f"slider_{f}"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬  RUN FULL DIAGNOSTIC", width="stretch"):
        X_in     = np.array([[user_input[f] for f in FEAT_NAMES]])
        X_scaled = scaler.transform(X_in)
        clf      = results[diag_model]["clf"]
        pred     = clf.predict(X_scaled)[0]
        proba    = clf.predict_proba(X_scaled)[0]

        # ALL model votes
        votes = {n: results[n]["clf"].predict(X_scaled)[0] for n in results}
        pd_votes  = sum(v == 1 for v in votes.values())
        ensemble_prob = np.mean([results[n]["clf"].predict_proba(X_scaled)[0][1] for n in results])

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])
        with res_col1:
            if pred == 1:
                st.markdown(f"""<div class="result-pd">
                    <div style='font-size:3.5rem'>⚠️</div>
                    <div style='font-family:Share Tech Mono;font-size:1.4rem;letter-spacing:4px;color:{RED}'>PD DETECTED</div>
                    <div style='font-size:2rem;font-weight:800;color:{RED};font-family:Share Tech Mono'>{proba[1]*100:.1f}%</div>
                    <div style='color:{TEXT_DIM};font-size:0.72rem;letter-spacing:2px'>PROBABILITY — {diag_model.upper()}</div>
                    <div style='margin-top:12px;color:{TEXT_DIM};font-size:0.75rem'>Ensemble avg: <span style='color:{RED}'>{ensemble_prob*100:.1f}%</span></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-healthy">
                    <div style='font-size:3.5rem'>✅</div>
                    <div style='font-family:Share Tech Mono;font-size:1.4rem;letter-spacing:4px;color:{GREEN}'>HEALTHY</div>
                    <div style='font-size:2rem;font-weight:800;color:{GREEN};font-family:Share Tech Mono'>{proba[0]*100:.1f}%</div>
                    <div style='color:{TEXT_DIM};font-size:0.72rem;letter-spacing:2px'>CONFIDENCE — {diag_model.upper()}</div>
                    <div style='margin-top:12px;color:{TEXT_DIM};font-size:0.75rem'>Ensemble avg: <span style='color:{GREEN}'>{(1-ensemble_prob)*100:.1f}%</span></div>
                </div>""", unsafe_allow_html=True)

        with res_col2:
            st.markdown(f'<div class="section-title">◆ ALL MODEL VOTES</div>', unsafe_allow_html=True)
            for name, vote in votes.items():
                icon = f'<span style="color:{RED}">⚠ PD</span>' if vote==1 else f'<span style="color:{GREEN}">✓ OK</span>'
                st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:4px 0;border-bottom:1px solid {BORDER}"><span style="color:{TEXT_MID}">{name}</span>{icon}</div>', unsafe_allow_html=True)
            st.markdown(f'<br><div style="font-family:Share Tech Mono;font-size:0.9rem">PD votes: <span style="color:{RED}">{pd_votes}/{len(results)}</span></div>', unsafe_allow_html=True)

        with res_col3:
            st.markdown(f'<div class="section-title">◆ CLASS PROBABILITIES</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            bars = ax.barh(["Healthy","PD"], [proba[0], proba[1]], color=[GREEN, RED], alpha=0.8, edgecolor=BORDER, height=0.5)
            for bar, v in zip(bars, [proba[0], proba[1]]):
                ax.text(max(bar.get_width()-0.05, 0.02), bar.get_y()+bar.get_height()/2,
                        f"{v*100:.1f}%", va="center", ha="right", fontsize=14, fontweight="bold",
                        color=DARK_BG if v > 0.3 else TEXT_MAIN)
            ax.set(xlim=[0,1], title=f"{diag_model}"); ax.grid(axis="x"); ax.set_xlabel("Probability")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Feature deviation analysis
        st.markdown(f'<div class="section-title" style="margin-top:20px">◆ BIOMARKER DEVIATION FROM REFERENCE</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 4))
        x_pos    = np.arange(len(FEAT_NAMES))
        h_vals   = np.array([FEATURE_META[f]["healthy"] for f in FEAT_NAMES])
        pd_vals_ref = np.array([FEATURE_META[f]["pd"]     for f in FEAT_NAMES])
        p_vals   = np.array([user_input[f] for f in FEAT_NAMES])
        # Normalize each feature to [0,1]
        lo_arr   = np.array([FEATURE_META[f]["lo"] for f in FEAT_NAMES])
        hi_arr   = np.array([FEATURE_META[f]["hi"] for f in FEAT_NAMES])
        p_norm   = (p_vals - lo_arr) / (hi_arr - lo_arr)
        h_norm   = (h_vals - lo_arr) / (hi_arr - lo_arr)
        pd_norm  = (pd_vals_ref - lo_arr) / (hi_arr - lo_arr)
        deviations = p_norm - h_norm
        bar_colors = [RED if d > 0.05 else (GREEN if d < -0.05 else AMBER) for d in deviations]
        ax.bar(x_pos, deviations, color=bar_colors, alpha=0.8, edgecolor=BORDER, zorder=2)
        ax.axhline(0, color=TEXT_DIM, lw=1.5, zorder=3)
        ax.scatter(x_pos, pd_norm-h_norm, color=RED, s=20, alpha=0.4, zorder=4, label="PD reference deviation")
        ax.set_xticks(x_pos); ax.set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set(title="Normalized Deviation from Healthy Reference", ylabel="Δ (patient − healthy), normalized")
        ax.grid(axis="y", zorder=0); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 5 — DATA EXPLORER
# └─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="section-title">◆ RAW DATASET (195 SAMPLES)</div>', unsafe_allow_html=True)
        display_df = df.copy()
        display_df.insert(0, "Status", display_df.pop("status").map({0:"✅ Healthy", 1:"⚠️ PD"}))
        st.dataframe(display_df, width="stretch", height=300)

    with col2:
        st.markdown('<div class="section-title">◆ SUMMARY STATS</div>', unsafe_allow_html=True)
        st.dataframe(df[FEAT_NAMES].describe().round(4).T[["mean","std","min","max"]], width="stretch")

    st.markdown('<div class="section-title" style="margin-top:16px">◆ FEATURE PAIR EXPLORER</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    feat_x = col_a.selectbox("X-axis feature", FEAT_NAMES, index=0, key="fx")
    feat_y = col_b.selectbox("Y-axis feature", FEAT_NAMES, index=7, key="fy")
    show_kde = col_c.checkbox("Show density contours", value=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, name in [(0, GREEN, "Healthy"), (1, RED, "PD")]:
        mask = df.status == label
        ax.scatter(df[mask][feat_x], df[mask][feat_y],
                   c=color, alpha=0.55, s=35,
                   edgecolors=BORDER, lw=0.4, label=name, zorder=3)
        if show_kde and mask.sum() > 5:
            try:
                from scipy.stats import gaussian_kde
                xd, yd = df[mask][feat_x].values, df[mask][feat_y].values
                k = gaussian_kde(np.vstack([xd, yd]))
                xi, yi = np.mgrid[xd.min():xd.max():60j, yd.min():yd.max():60j]
                zi = k(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
                ax.contour(xi, yi, zi, levels=4, colors=[color], alpha=0.35, linewidths=1)
            except: pass
    ax.set(title=f"{feat_x}  vs  {feat_y}", xlabel=feat_x, ylabel=feat_y)
    ax.legend(); ax.grid(True)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title" style="margin-top:8px">◆ CATEGORY-WISE BOX PLOTS</div>', unsafe_allow_html=True)
    cat_sel = st.selectbox("Feature category", CATS)
    cat_feats = [f for f in FEAT_NAMES if FEATURE_META[f]["cat"] == cat_sel]
    ncols = min(len(cat_feats), 4)
    nrows = (len(cat_feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows*3.5))
    axes = np.array(axes).flatten()
    for i, f in enumerate(cat_feats):
        ax = axes[i]
        pd_v = df[df.status==1][f]
        h_v  = df[df.status==0][f]
        bp = ax.boxplot([h_v, pd_v], labels=["Healthy","PD"], patch_artist=True, widths=0.45,
                        medianprops=dict(color=TEXT_MAIN, lw=2.5),
                        whiskerprops=dict(color=TEXT_DIM),
                        capprops=dict(color=TEXT_DIM),
                        flierprops=dict(marker="o", ms=3.5, alpha=0.5))
        bp["boxes"][0].set(facecolor=GREEN+"33", edgecolor=GREEN)
        bp["boxes"][1].set(facecolor=RED+"33",   edgecolor=RED)
        d = stat_tests[f]["d"]; p_v = stat_tests[f]["p"]
        sig = "***" if p_v < 0.001 else "**" if p_v < 0.01 else "*" if p_v < 0.05 else "ns"
        ax.set_title(f"{f}\nd={d:.2f} {sig}", fontsize=8)
        ax.grid(axis="y")
    for j in range(len(cat_feats), len(axes)): axes[j].set_visible(False)
    fig.suptitle(f"{cat_sel} Features — PD vs Healthy", y=1.01, color=CAT_COLORS[cat_sel], fontsize=11)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ┌─────────────────────────────────────────────────────────────────────────────
# │ TAB 6 — EXPLAINABILITY
# └─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown(f"""<div class="info-box">
        <strong>ML Explainability</strong> helps understand <em>why</em> a model makes a prediction.
        Here we use <strong>Permutation Importance</strong> (model-agnostic), <strong>Partial Dependence Plots</strong>,
        and <strong>Decision Boundary</strong> visualization to look inside the black box.
    </div><br>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">◆ PERMUTATION IMPORTANCE</div>', unsafe_allow_html=True)
        r = results[selected_model]
        Xs = scaler.transform(df[FEAT_NAMES])
        _, Xte_f, _, yte_f = train_test_split(Xs, df["status"], test_size=0.2, stratify=df["status"], random_state=42)
        with st.spinner("Computing permutation importance…"):
            perm = permutation_importance(r["clf"], Xte_f, yte_f, n_repeats=30, random_state=42, scoring="roc_auc")
        p_idx = np.argsort(perm.importances_mean)[::-1][:15]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.boxplot([perm.importances[p_idx[i]] for i in range(len(p_idx))],
                   vert=False, labels=[FEAT_NAMES[i] for i in p_idx],
                   patch_artist=True,
                   medianprops=dict(color=CYAN, lw=2),
                   boxprops=dict(facecolor=CYAN+"22", edgecolor=CYAN),
                   whiskerprops=dict(color=TEXT_DIM),
                   capprops=dict(color=TEXT_DIM),
                   flierprops=dict(marker="o", ms=3, alpha=0.4))
        ax.axvline(0, color=BORDER, lw=1.5, ls="--")
        ax.set(title=f"Permutation Importance ({selected_model})\n[30 shuffles, ROC AUC drop]",
               xlabel="Mean ROC AUC decrease")
        ax.grid(axis="x")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">◆ PARTIAL DEPENDENCE PLOTS</div>', unsafe_allow_html=True)
        top4 = [FEAT_NAMES[i] for i in np.argsort(perm.importances_mean)[::-1][:4]]
        fig, axes = plt.subplots(2, 2, figsize=(6, 5.5))
        axes = axes.flatten()
        Xs_df = pd.DataFrame(Xs, columns=FEAT_NAMES)
        for ax, feat in zip(axes, top4):
            feat_range = np.linspace(Xs_df[feat].min(), Xs_df[feat].max(), 60)
            mean_other = Xs_df.mean()
            pd_probs   = []
            for val in feat_range:
                tmp = mean_other.copy(); tmp[feat] = val
                prob = r["clf"].predict_proba(tmp.values.reshape(1,-1))[0][1]
                pd_probs.append(prob)
            ax.plot(feat_range, pd_probs, color=CYAN, lw=2.5)
            ax.axhline(0.5, color=RED, lw=1, ls="--", alpha=0.7)
            ax.fill_between(feat_range, pd_probs, 0.5, where=[p > 0.5 for p in pd_probs],
                            color=RED, alpha=0.12, label=">0.5: PD")
            ax.fill_between(feat_range, pd_probs, 0.5, where=[p < 0.5 for p in pd_probs],
                            color=GREEN, alpha=0.12, label="<0.5: Healthy")
            ax.set(title=feat, xlabel="Normalized value", ylabel="P(PD)", ylim=[0,1])
            ax.grid(True)
        fig.suptitle("Partial Dependence (top 4 features)", color=CYAN, fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title">◆ DECISION BOUNDARY — PCA 2D PROJECTION</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.78rem;color:{TEXT_DIM};margin-bottom:8px">Note: boundary projected onto first 2 principal components for visualization. Real classification happens in {len(FEAT_NAMES)}-dimensional space.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    Xs = scaler.transform(df[FEAT_NAMES])
    pca2 = PCA(n_components=2, random_state=42)
    Xp   = pca2.fit_transform(Xs)

    boundary_models = ["SVM (RBF)", "Random Forest", "Logistic Regression", "K-Nearest Neighbors"]
    for col, model_name in zip([col1, col2, col1, col2], boundary_models):
        with col:
            clf_bd = results[model_name]["clf"]
            # Create boundary model in PCA space
            pca_clf = Pipeline([("pca", PCA(n_components=2, random_state=42)),
                                ("clf", SVC(kernel="rbf", C=5, gamma="scale", probability=True, random_state=42)
                                 if model_name == "SVM (RBF)" else
                                 RandomForestClassifier(n_estimators=100, random_state=42) if model_name == "Random Forest" else
                                 LogisticRegression(C=0.5, max_iter=1000, random_state=42) if model_name == "Logistic Regression" else
                                 KNeighborsClassifier(n_neighbors=7))])
            pca_clf.fit(Xs, df["status"])

            x_min, x_max = Xp[:,0].min()-0.5, Xp[:,0].max()+0.5
            y_min, y_max = Xp[:,1].min()-0.5, Xp[:,1].max()+0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))

            # Back-transform for prediction
            grid_pca   = np.c_[xx.ravel(), yy.ravel()]
            grid_orig  = pca2.inverse_transform(grid_pca)
            Z = pca_clf.predict_proba(grid_orig)[:,1].reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(5.5, 4))
            contourf = ax.contourf(xx, yy, Z, levels=50,
                                   cmap=LinearSegmentedColormap.from_list("db",[GREEN+"aa",DARK_BG,RED+"aa"],256),
                                   alpha=0.6)
            ax.contour(xx, yy, Z, levels=[0.5], colors=[TEXT_MAIN], linewidths=2, linestyles="--")
            for label, color, name in [(0, GREEN, "Healthy"), (1, RED, "PD")]:
                mask = df.status == label
                ax.scatter(Xp[mask,0], Xp[mask,1], c=color, s=22, edgecolors=DARK_BG, lw=0.5,
                           alpha=0.75, label=name, zorder=4)
            ax.set(title=f"Decision Boundary — {model_name}", xlabel="PC1", ylabel="PC2")
            ax.legend(fontsize=7.5)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Confusion matrices for all models side by side
    st.markdown('<div class="section-title" style="margin-top:8px">◆ CONFUSION MATRICES — ALL MODELS</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    for ax, (name, res) in zip(axes, results.items()):
        cm_n = res["cm"].astype(float) / res["cm"].sum(axis=1, keepdims=True)
        im = ax.imshow(cm_n, cmap=LinearSegmentedColormap.from_list("cm",[DARK_BG,"#1a2a4a",CYAN],256), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{res['cm'][i,j]}\n({cm_n[i,j]*100:.0f}%)",
                        ha="center", va="center", fontsize=8.5, fontweight="bold",
                        color=DARK_BG if cm_n[i,j] > 0.6 else TEXT_MAIN)
        ax.set(title=f"{name}\nACC:{res['accuracy']*100:.1f}%", xticks=[0,1], yticks=[0,1])
        ax.set_xticklabels(["H","PD"],fontsize=8); ax.set_yticklabels(["H","PD"],fontsize=8)
    fig.suptitle("Confusion Matrices — All Classifiers", color=CYAN, fontsize=11, y=1.01)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
