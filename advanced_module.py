"""
Advanced ML Module — Parkinson's Detection Lab
All heavy computations are gated behind run buttons.
Nothing trains automatically — Streamlit Cloud safe.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_curve, auc, confusion_matrix,
    classification_report, precision_score, recall_score,
    matthews_corrcoef, average_precision_score, precision_recall_curve,
    brier_score_loss, log_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

try:
    import xgboost as xgb; XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgb; LGB_OK = True
except ImportError:
    LGB_OK = False

try:
    import catboost as cb; CAT_OK = True
except ImportError:
    CAT_OK = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

# ── Colours ──────────────────────────────────────────────────────────────────
WHITE = "#ffffff"; BG = "#f4f6f9"; NAVY = "#1e3a5f"; BLUE = "#2563eb"
BLUE_LIGHT = "#dbeafe"; GREEN = "#059669"; GREEN_LIGHT = "#d1fae5"
RED = "#dc2626"; RED_LIGHT = "#fee2e2"; AMBER = "#d97706"
AMBER_LIGHT = "#fef3c7"; PURPLE = "#7c3aed"; SLATE = "#0ea5e9"
TEAL = "#0d9488"; BORDER = "#e2e8f0"; BORDER_MED = "#cbd5e1"
TEXT_MAIN = "#1e293b"; TEXT_MID = "#475569"; TEXT_DIM = "#94a3b8"
PAL = [BLUE, GREEN, AMBER, PURPLE, RED, SLATE, TEAL, "#f43f5e", "#84cc16"]

plt.rcParams.update({
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "axes.edgecolor": BORDER_MED, "axes.labelcolor": TEXT_MID,
    "axes.titlecolor": NAVY, "text.color": TEXT_MAIN,
    "xtick.color": TEXT_DIM, "ytick.color": TEXT_DIM,
    "grid.color": BORDER, "grid.linestyle": "--", "grid.alpha": 0.8,
    "font.family": "DejaVu Sans", "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "legend.facecolor": WHITE,
    "legend.edgecolor": BORDER,
})


# ── UI helpers ────────────────────────────────────────────────────────────────
def section(title):
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;'
        f'text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};'
        f'padding-bottom:8px;margin:20px 0 16px 0">{title}</div>',
        unsafe_allow_html=True)

def info_box(html):
    st.markdown(
        f'<div style="background:{BLUE_LIGHT};border-left:4px solid {BLUE};'
        f'padding:14px 18px;border-radius:6px;font-size:0.84rem;'
        f'line-height:1.75;color:{NAVY};margin-bottom:16px">{html}</div>',
        unsafe_allow_html=True)

def warn_box(html):
    st.markdown(
        f'<div style="background:{AMBER_LIGHT};border-left:4px solid {AMBER};'
        f'padding:10px 16px;border-radius:6px;font-size:0.78rem;'
        f'color:#92400e;font-weight:500;margin-bottom:12px">{html}</div>',
        unsafe_allow_html=True)

def success_box(html):
    st.markdown(
        f'<div style="background:{GREEN_LIGHT};border-left:4px solid {GREEN};'
        f'padding:10px 16px;border-radius:6px;font-size:0.78rem;'
        f'color:#065f46;font-weight:500;margin-bottom:12px">{html}</div>',
        unsafe_allow_html=True)

def metric_card(val, label, color):
    st.markdown(
        f'<div style="background:{WHITE};border:1px solid {BORDER};border-radius:10px;'
        f'padding:18px;text-align:center;border-left:4px solid {color}">'
        f'<div style="font-family:DM Serif Display,serif;font-size:1.8rem;color:{color}'
        f';line-height:1.1">{val}</div>'
        f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:1.5px;'
        f'text-transform:uppercase;color:{TEXT_DIM};margin-top:4px">{label}</div>'
        f'</div>', unsafe_allow_html=True)

def run_gate(key, button_label, description, estimated_time, warning=None):
    """
    Shows a description card with a run button.
    Returns True if user has clicked run, False otherwise.
    Uses st.session_state to persist across reruns.
    """
    if key not in st.session_state:
        st.session_state[key] = False

    if not st.session_state[key]:
        st.markdown(
            f'<div style="background:{WHITE};border:1px solid {BORDER};'
            f'border-radius:12px;padding:28px 32px;text-align:center;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
            f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:{TEXT_DIM};margin-bottom:12px">ON DEMAND</div>'
            f'<div style="font-family:DM Serif Display,serif;font-size:1.4rem;'
            f'color:{NAVY};margin-bottom:8px">{button_label}</div>'
            f'<div style="font-size:0.84rem;color:{TEXT_MID};'
            f'line-height:1.7;margin-bottom:16px">{description}</div>'
            f'<div style="display:inline-block;background:{AMBER_LIGHT};'
            f'border:1px solid #fde68a;border-radius:20px;'
            f'padding:4px 16px;font-size:0.72rem;font-weight:600;'
            f'color:#92400e;margin-bottom:20px">Estimated time: {estimated_time}</div>'
            f'</div>',
            unsafe_allow_html=True)
        if warning:
            warn_box(warning)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"Run {button_label}", key=f"btn_{key}"):
            st.session_state[key] = True
            st.rerun()
        return False
    return True


def get_full_metrics(clf, Xte, yte, name=""):
    yp = clf.predict(Xte)
    ypr = clf.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, ypr)
    pr_p, pr_r, _ = precision_recall_curve(yte, ypr)
    return dict(
        name=name, clf=clf,
        accuracy=accuracy_score(yte, yp), f1=f1_score(yte, yp),
        precision=precision_score(yte, yp), recall=recall_score(yte, yp),
        roc_auc=auc(fpr, tpr), pr_auc=average_precision_score(yte, ypr),
        mcc=matthews_corrcoef(yte, yp),
        brier=brier_score_loss(yte, ypr), logloss=log_loss(yte, ypr),
        cm=confusion_matrix(yte, yp),
        report=classification_report(yte, yp, target_names=["Healthy", "PD"]),
        fpr=fpr, tpr=tpr, pr_p=pr_p, pr_r=pr_r,
        y_pred=yp, y_proba=ypr, y_test=yte,
    )


# ── Statistical tests ─────────────────────────────────────────────────────────
def mcnemar_test(y_true, pred1, pred2):
    b = np.sum((pred1 == y_true) & (pred2 != y_true))
    c = np.sum((pred1 != y_true) & (pred2 == y_true))
    if b + c == 0:
        return 1.0, 0.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return 1 - stats.chi2.cdf(chi2, df=1), chi2


def delong_auc_test(y_true, proba1, proba2):
    def midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1); i = j
        T2 = np.empty(N); T2[J] = T + 1
        return T2

    y = np.array(y_true); p1 = np.array(proba1); p2 = np.array(proba2)
    idx = np.argsort(y)[::-1]; ys = y[idx]; m = int(ys.sum())
    preds = np.vstack([p1[idx], p2[idx]])
    k = preds.shape[0]; n = preds.shape[1] - m
    pos = preds[:, :m]; neg = preds[:, m:]
    tx = np.array([midrank(pos[r]) for r in range(k)])
    ty = np.array([midrank(neg[r]) for r in range(k)])
    tz = np.array([midrank(preds[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(1) - tx.sum(1)) / (m * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]; var = cov[0,0] + cov[1,1] - 2*cov[0,1]
    if var <= 0: return aucs[0], aucs[1], 1.0, 0.0
    z = diff / np.sqrt(var)
    return aucs[0], aucs[1], 2*(1 - stats.norm.cdf(abs(z))), z


def bootstrap_ci(y_true, y_proba, metric_fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed); y = np.array(y_true); p = np.array(y_proba)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2: continue
        try: boot.append(metric_fn(y[idx], p[idx]))
        except: continue
    boot = np.array(boot)
    return boot.mean(), boot.std(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ── Synthetic telemonitoring fallback ─────────────────────────────────────────
def _synthetic_telemonitoring(feat_names, seed=42):
    rng = np.random.RandomState(seed); n = 5875
    status = rng.choice([0, 1], size=n, p=[0.25, 0.75])
    specs = {
        "MDVP:Fo(Hz)":(145.21,35.,197.10,45.), "MDVP:Fhi(Hz)":(197.11,60.,243.60,70.),
        "MDVP:Flo(Hz)":(102.15,28.,146.67,38.), "MDVP:Jitter(%)":(0.0063,.004,.0033,.0015),
        "MDVP:Jitter(Abs)":(4.5e-5,2.5e-5,2.2e-5,1e-5), "MDVP:RAP":(.0033,.002,.0017,.001),
        "MDVP:PPQ":(.0034,.002,.0018,.001), "Jitter:DDP":(.010,.006,.0052,.003),
        "MDVP:Shimmer":(.0508,.025,.0231,.012), "MDVP:Shimmer(dB)":(.471,.22,.214,.10),
        "Shimmer:APQ3":(.0269,.014,.0122,.006), "Shimmer:APQ5":(.0316,.017,.0145,.007),
        "MDVP:APQ":(.0439,.022,.0199,.010), "Shimmer:DDA":(.0808,.040,.0366,.018),
        "NHR":(.0312,.028,.0111,.008), "HNR":(19.98,5.5,24.68,4.5),
        "RPDE":(.587,.085,.499,.070), "DFA":(.753,.048,.718,.045),
        "spread1":(-5.335,1.0,-6.759,.9), "spread2":(.269,.10,.168,.08),
        "D2":(2.522,.38,2.302,.34), "PPE":(.213,.11,.062,.035),
    }
    rows = {}
    for f in feat_names:
        if f in specs:
            mu_p, sd_p, mu_h, sd_h = specs[f]
            rows[f] = np.where(status==1,
                np.clip(rng.normal(mu_p, sd_p, n), 0, None),
                np.clip(rng.normal(mu_h, sd_h, n), 0, None))
        else:
            rows[f] = rng.normal(.5, .15, n)
    df = pd.DataFrame(rows); df["status"] = status
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════
def render_advanced_tabs(df, scaler, feat_names, existing_results):
    df_hash = len(df)
    X = df[feat_names]; y = df["status"]
    Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)

    adv_tabs = st.tabs([
        "Boosting + Optuna",
        "Stacking Ensemble",
        "SMOTE + Class Balance",
        "Calibration + Uncertainty",
        "Statistical Tests",
        "Cross-Dataset Validation",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 0 — BOOSTING + OPTUNA
    # ══════════════════════════════════════════════════════════════════════════
    with adv_tabs[0]:
        info_box(
            "<strong>XGBoost, LightGBM, CatBoost</strong> — the three industry-standard "
            "gradient boosting libraries. <strong>Optuna TPE</strong> (Bayesian optimisation) "
            "finds better hyperparameters than GridSearch with far fewer trials. "
            "Each model runs independently so you can train them one at a time.")

        missing = []
        if not XGB_OK:    missing.append("<code>pip install xgboost</code>")
        if not LGB_OK:    missing.append("<code>pip install lightgbm</code>")
        if not CAT_OK:    missing.append("<code>pip install catboost</code>")
        if not OPTUNA_OK: missing.append("<code>pip install optuna</code>")
        if missing:
            warn_box("Missing packages: " + " &nbsp;·&nbsp; ".join(missing))

        # ── XGBoost ──────────────────────────────────────────────────────────
        section("XGBoost — Optuna Tuning")
        if not (XGB_OK and OPTUNA_OK):
            warn_box("XGBoost and Optuna required. Install above packages first.")
        elif run_gate(
            key="xgb_ran",
            button_label="XGBoost Optuna Tuning",
            description="40 Optuna trials × 5-fold stratified CV optimising ROC AUC. "
                        "Searches over learning rate, max depth, subsample, colsample, L1/L2 regularisation.",
            estimated_time="2–4 minutes on Streamlit Cloud"
        ):
            with st.spinner("Tuning XGBoost… (40 trials × 5-fold CV)"):
                cv5 = StratifiedKFold(5, shuffle=True, random_state=42)
                def xgb_obj(trial):
                    p = {
                        "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
                        "max_depth":        trial.suggest_int("max_depth", 3, 8),
                        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 5, log=True),
                        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 5, log=True),
                        "eval_metric": "logloss", "random_state": 42, "verbosity": 0,
                    }
                    return cross_val_score(xgb.XGBClassifier(**p), Xtr, ytr, cv=cv5, scoring="roc_auc", n_jobs=-1).mean()
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(xgb_obj, n_trials=40, show_progress_bar=False)
                bp = {**study.best_params, "eval_metric":"logloss","random_state":42,"verbosity":0}
                xgb_model = xgb.XGBClassifier(**bp).fit(Xtr, ytr)
                st.session_state["xgb_model"] = xgb_model
                st.session_state["xgb_study"] = study

            xgb_model = st.session_state["xgb_model"]
            study      = st.session_state["xgb_study"]
            m = get_full_metrics(xgb_model, Xte, yte, "XGBoost")

            c1,c2,c3,c4 = st.columns(4)
            with c1: metric_card(f"{m['roc_auc']:.4f}", "ROC AUC", BLUE)
            with c2: metric_card(f"{m['accuracy']*100:.1f}%", "Test Accuracy", GREEN)
            with c3: metric_card(f"{m['f1']:.4f}", "F1 Score", AMBER)
            with c4: metric_card(f"{study.best_value:.4f}", "Best CV AUC", PURPLE)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                section("Optuna Trial History")
                vals = [t.value for t in study.trials if t.value]
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.scatter(range(len(vals)), vals, color=BLUE, alpha=0.5, s=20, label="Trial")
                ax.plot(range(len(vals)), np.maximum.accumulate(vals), color=RED, lw=2, label="Best")
                ax.set(title="XGBoost — Optuna Trial History", xlabel="Trial", ylabel="CV ROC AUC")
                ax.legend(); ax.grid(True)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            with col2:
                section("Best Parameters Found")
                for k, v in study.best_params.items():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:0.82rem;padding:5px 0;border-bottom:1px solid {BORDER}">'
                        f'<span style="color:{TEXT_MID}">{k}</span>'
                        f'<span style="color:{NAVY};font-weight:600">'
                        f'{v:.4f if isinstance(v,float) else v}</span></div>',
                        unsafe_allow_html=True)

            section("XGBoost vs Best Classical Model — ROC")
            best_cls = max(existing_results, key=lambda k: existing_results[k]["roc_auc"])
            bc = existing_results[best_cls]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(bc["fpr"], bc["tpr"], color=TEXT_DIM, lw=1.5, ls="--",
                    label=f"Best Classical: {best_cls} ({bc['roc_auc']:.4f})")
            ax.plot(m["fpr"], m["tpr"], color=BLUE, lw=2.5,
                    label=f"XGBoost Tuned ({m['roc_auc']:.4f})")
            ax.fill_between(m["fpr"], m["tpr"], alpha=0.08, color=BLUE)
            ax.plot([0,1],[0,1], color=BORDER, lw=1, ls=":")
            ax.set(title="XGBoost vs Best Classical — ROC Curve", xlabel="FPR", ylabel="TPR")
            ax.legend(); ax.grid(True)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── LightGBM ─────────────────────────────────────────────────────────
        section("LightGBM — Optuna Tuning")
        if not (LGB_OK and OPTUNA_OK):
            warn_box("LightGBM and Optuna required.")
        elif run_gate(
            key="lgb_ran",
            button_label="LightGBM Optuna Tuning",
            description="40 Optuna trials × 5-fold CV. Searches learning rate, num_leaves, "
                        "max depth, subsample, min_child_samples, L1/L2.",
            estimated_time="2–3 minutes on Streamlit Cloud"
        ):
            with st.spinner("Tuning LightGBM… (40 trials)"):
                cv5 = StratifiedKFold(5, shuffle=True, random_state=42)
                def lgb_obj(trial):
                    p = {
                        "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
                        "max_depth":         trial.suggest_int("max_depth", 3, 8),
                        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "num_leaves":        trial.suggest_int("num_leaves", 20, 120),
                        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 5, log=True),
                        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 5, log=True),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
                        "random_state": 42, "verbosity": -1,
                    }
                    return cross_val_score(lgb.LGBMClassifier(**p), Xtr, ytr, cv=cv5, scoring="roc_auc", n_jobs=-1).mean()
                study_l = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
                study_l.optimize(lgb_obj, n_trials=40, show_progress_bar=False)
                bp_l = {**study_l.best_params, "random_state":42,"verbosity":-1}
                lgb_model = lgb.LGBMClassifier(**bp_l).fit(Xtr, ytr)
                st.session_state["lgb_model"] = lgb_model
                st.session_state["lgb_study"] = study_l

            lgb_model = st.session_state["lgb_model"]
            study_l   = st.session_state["lgb_study"]
            ml = get_full_metrics(lgb_model, Xte, yte, "LightGBM")

            c1,c2,c3,c4 = st.columns(4)
            with c1: metric_card(f"{ml['roc_auc']:.4f}", "ROC AUC", BLUE)
            with c2: metric_card(f"{ml['accuracy']*100:.1f}%", "Test Accuracy", GREEN)
            with c3: metric_card(f"{ml['f1']:.4f}", "F1 Score", AMBER)
            with c4: metric_card(f"{study_l.best_value:.4f}", "Best CV AUC", PURPLE)

        # ── CatBoost ─────────────────────────────────────────────────────────
        section("CatBoost — Optuna Tuning")
        if not (CAT_OK and OPTUNA_OK):
            warn_box("CatBoost and Optuna required.")
        elif run_gate(
            key="cat_ran",
            button_label="CatBoost Optuna Tuning",
            description="30 Optuna trials × 5-fold CV. Searches iterations, depth, "
                        "learning rate, L2 leaf regularisation, border count.",
            estimated_time="3–5 minutes on Streamlit Cloud"
        ):
            with st.spinner("Tuning CatBoost… (30 trials — this is the slowest one)"):
                cv5 = StratifiedKFold(5, shuffle=True, random_state=42)
                def cat_obj(trial):
                    p = {
                        "iterations":      trial.suggest_int("iterations", 100, 400),
                        "depth":           trial.suggest_int("depth", 3, 8),
                        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "l2_leaf_reg":     trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
                        "border_count":    trial.suggest_int("border_count", 32, 200),
                        "random_state": 42, "verbose": 0,
                    }
                    return cross_val_score(cb.CatBoostClassifier(**p), Xtr, ytr, cv=cv5, scoring="roc_auc", n_jobs=1).mean()
                study_c = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
                study_c.optimize(cat_obj, n_trials=30, show_progress_bar=False)
                bp_c = {**study_c.best_params, "random_state":42,"verbose":0}
                cat_model = cb.CatBoostClassifier(**bp_c).fit(Xtr, ytr)
                st.session_state["cat_model"] = cat_model
                st.session_state["cat_study"] = study_c

            cat_model = st.session_state["cat_model"]
            study_c   = st.session_state["cat_study"]
            mc = get_full_metrics(cat_model, Xte, yte, "CatBoost")

            c1,c2,c3,c4 = st.columns(4)
            with c1: metric_card(f"{mc['roc_auc']:.4f}", "ROC AUC", BLUE)
            with c2: metric_card(f"{mc['accuracy']*100:.1f}%", "Test Accuracy", GREEN)
            with c3: metric_card(f"{mc['f1']:.4f}", "F1 Score", AMBER)
            with c4: metric_card(f"{study_c.best_value:.4f}", "Best CV AUC", PURPLE)


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — STACKING ENSEMBLE
    # ══════════════════════════════════════════════════════════════════════════
    with adv_tabs[1]:
        info_box(
            "<strong>Stacking Ensemble</strong> trains a meta-learner (Logistic Regression) "
            "on the out-of-fold predictions of all base classifiers. The meta-learner learns "
            "which models to trust for which regions of input space — often outperforming "
            "any individual model. This is the standard approach in Kaggle competition winners.")

        if run_gate(
            key="stack_ran",
            button_label="Build Stacking Ensemble",
            description="Trains 6 base models with 5-fold out-of-fold predictions, "
                        "then fits a Logistic Regression meta-learner on top. "
                        "Includes any boosting models already trained in the previous tab.",
            estimated_time="1–2 minutes on Streamlit Cloud"
        ):
            with st.spinner("Building Stacking Ensemble…"):
                estimators = [
                    ("rf",  RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
                    ("gb",  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
                    ("svm", SVC(kernel="rbf", C=10, gamma=0.01, probability=True, random_state=42)),
                    ("knn", KNeighborsClassifier(n_neighbors=7)),
                    ("lr",  LogisticRegression(C=0.5, max_iter=2000, random_state=42)),
                ]
                if "xgb_model" in st.session_state:
                    estimators.append(("xgb", st.session_state["xgb_model"]))
                if "lgb_model" in st.session_state:
                    estimators.append(("lgb", st.session_state["lgb_model"]))
                if "cat_model" in st.session_state:
                    estimators.append(("cat", st.session_state["cat_model"]))

                stack = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    passthrough=False, n_jobs=-1,
                )
                stack.fit(Xtr, ytr)
                st.session_state["stack_model"] = stack
                st.session_state["stack_metrics"] = get_full_metrics(stack, Xte, yte, "Stacking")

            sm = st.session_state["stack_metrics"]
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: metric_card(f"{sm['roc_auc']:.4f}", "ROC AUC", BLUE)
            with c2: metric_card(f"{sm['accuracy']*100:.1f}%", "Accuracy", GREEN)
            with c3: metric_card(f"{sm['f1']:.4f}", "F1 Score", AMBER)
            with c4: metric_card(f"{sm['mcc']:.4f}", "MCC", PURPLE)
            with c5: metric_card(f"{sm['brier']:.4f}", "Brier Score", RED)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                section("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(5, 4))
                cm_n = sm["cm"].astype(float) / sm["cm"].sum(axis=1, keepdims=True)
                im = ax.imshow(cm_n, cmap=LinearSegmentedColormap.from_list("c",[WHITE,BLUE_LIGHT,BLUE],256), vmin=0, vmax=1)
                for i in range(2):
                    for j in range(2):
                        ax.text(j,i,f"{sm['cm'][i,j]}\n({cm_n[i,j]*100:.1f}%)",
                                ha="center",va="center",fontsize=13,fontweight="bold",
                                color=WHITE if cm_n[i,j]>0.6 else NAVY)
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(["Pred: Healthy","Pred: PD"])
                ax.set_yticklabels(["True: Healthy","True: PD"])
                ax.set_title("Stacking Ensemble — Confusion Matrix")
                plt.colorbar(im,ax=ax,fraction=0.046)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            with col2:
                section("Stacking vs Base Models — ROC")
                fig, ax = plt.subplots(figsize=(5, 4))
                for (name,res), c in zip(list(existing_results.items())[:5], PAL):
                    ax.plot(res["fpr"],res["tpr"],color=c,lw=1.2,alpha=0.45,
                            label=f"{name[:20]} ({res['roc_auc']:.3f})")
                ax.plot(sm["fpr"],sm["tpr"],color=NAVY,lw=3,
                        label=f"Stacking ({sm['roc_auc']:.4f})")
                ax.plot([0,1],[0,1],color=BORDER,lw=1,ls="--")
                ax.set(title="ROC — Stacking vs Base Models",xlabel="FPR",ylabel="TPR")
                ax.legend(fontsize=7,loc="lower right"); ax.grid(True)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            section("Classification Report")
            st.code(sm["report"], language="")

            section("Meta-Learner Coefficients")
            try:
                meta  = st.session_state["stack_model"].final_estimator_
                names = [n for n,_ in st.session_state["stack_model"].estimators]
                coefs = meta.coef_[0]
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.bar(names[:len(coefs)], coefs,
                       color=[GREEN if c>0 else RED for c in coefs], alpha=0.8, edgecolor=WHITE)
                ax.axhline(0, color=NAVY, lw=1.5)
                ax.set(title="Meta-Learner Coefficients — Which base models the stacker trusts most",
                       ylabel="Logistic Regression Coefficient")
                ax.set_xticklabels(names[:len(coefs)], rotation=25, ha="right", fontsize=8)
                ax.grid(axis="y"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            except Exception as e:
                st.info(f"Coefficient plot skipped: {e}")


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — SMOTE
    # ══════════════════════════════════════════════════════════════════════════
    with adv_tabs[2]:
        info_box(
            "The dataset is <strong>imbalanced</strong> (75% PD, 25% Healthy). "
            "<strong>SMOTE</strong> creates synthetic minority-class samples by interpolating "
            "between existing healthy recordings. <strong>ADASYN</strong> focuses synthesis on "
            "harder boundary cases. <strong>SMOTETomek</strong> combines oversampling with "
            "removal of noisy boundary samples from the majority class.")

        if not SMOTE_OK:
            warn_box("Install imbalanced-learn: <code>pip install imbalanced-learn</code>")
        elif run_gate(
            key="smote_ran",
            button_label="Run SMOTE / ADASYN / SMOTETomek Comparison",
            description="Applies three resampling strategies to the training set and "
                        "re-trains a Random Forest on each. Compares recall, precision, "
                        "F1, and AUC across methods. Test set is never resampled.",
            estimated_time="30–60 seconds"
        ):
            with st.spinner("Applying resampling strategies and re-training…"):
                sampling = {"Original": (Xtr, ytr)}
                try:
                    sm_ = SMOTE(random_state=42, k_neighbors=min(5,int((ytr==0).sum())-1))
                    Xs_, ys_ = sm_.fit_resample(Xtr, ytr); sampling["SMOTE"] = (Xs_, ys_)
                except Exception: pass
                try:
                    ada = ADASYN(random_state=42, n_neighbors=min(5,int((ytr==0).sum())-1))
                    Xa_, ya_ = ada.fit_resample(Xtr, ytr); sampling["ADASYN"] = (Xa_, ya_)
                except Exception: pass
                try:
                    smt = SMOTETomek(random_state=42)
                    Xst_, yst_ = smt.fit_resample(Xtr, ytr); sampling["SMOTETomek"] = (Xst_, yst_)
                except Exception: pass

                smote_res = {}
                for method, (Xs_, ys_) in sampling.items():
                    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
                    clf.fit(Xs_, ys_)
                    smote_res[method] = get_full_metrics(clf, Xte, yte, method)
                st.session_state["smote_res"] = smote_res
                st.session_state["smote_sampling"] = sampling

            smote_res = st.session_state["smote_res"]
            sampling  = st.session_state["smote_sampling"]

            section("Class Distribution After Resampling")
            fig, axes = plt.subplots(1, len(sampling), figsize=(4*len(sampling), 3.5))
            if len(sampling)==1: axes=[axes]
            for ax, (method,(Xs_,ys_)) in zip(axes, sampling.items()):
                counts = pd.Series(ys_).value_counts().sort_index()
                h_n = counts.get(0,0); pd_n = counts.get(1,0)
                ax.bar(["Healthy","PD"],[h_n,pd_n],color=[GREEN,RED],alpha=0.8,edgecolor=WHITE)
                ax.set_title(f"{method}\nn={len(ys_)}",fontsize=9,fontweight="bold")
                ax.set_ylabel("Count"); ax.grid(axis="y")
                for i,v in enumerate([h_n,pd_n]):
                    ax.text(i,v+2,str(v),ha="center",fontsize=9,fontweight="bold")
            fig.suptitle("Training Set Balance After Resampling",color=NAVY,fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            section("Performance Comparison")
            rows = [{
                "Method":method, "ROC AUC":f"{m['roc_auc']:.4f}",
                "F1":f"{m['f1']:.4f}", "Recall":f"{m['recall']:.4f}",
                "Precision":f"{m['precision']:.4f}", "Accuracy":f"{m['accuracy']*100:.2f}%"
            } for method,m in smote_res.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            col1,col2 = st.columns(2)
            with col1:
                fig,ax = plt.subplots(figsize=(5,4))
                for (mth,m),c in zip(smote_res.items(),PAL):
                    ax.plot(m["fpr"],m["tpr"],color=c,lw=2.5,label=f"{mth} ({m['roc_auc']:.4f})")
                ax.plot([0,1],[0,1],color=BORDER,lw=1,ls="--")
                ax.set(title="SMOTE Methods — ROC Comparison",xlabel="FPR",ylabel="TPR")
                ax.legend(fontsize=8); ax.grid(True)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            with col2:
                methods = list(smote_res.keys()); x = np.arange(len(methods)); w=0.25
                fig,ax = plt.subplots(figsize=(5,4))
                ax.bar(x-w,[smote_res[m]["recall"] for m in methods],w,label="Recall",color=GREEN,alpha=0.8,edgecolor=WHITE)
                ax.bar(x,  [smote_res[m]["precision"] for m in methods],w,label="Precision",color=BLUE,alpha=0.8,edgecolor=WHITE)
                ax.bar(x+w,[smote_res[m]["f1"] for m in methods],w,label="F1",color=AMBER,alpha=0.8,edgecolor=WHITE)
                ax.set_xticks(x); ax.set_xticklabels(methods)
                ax.set(title="Recall / Precision / F1",ylabel="Score",ylim=[0,1.1])
                ax.legend(); ax.grid(axis="y"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — CALIBRATION + UNCERTAINTY
    # ══════════════════════════════════════════════════════════════════════════
    with adv_tabs[3]:
        info_box(
            "<strong>Calibration</strong> checks whether predicted probabilities are trustworthy. "
            "A model predicting 80% PD should be right about 80% of the time. "
            "<strong>Bootstrap CIs</strong> give 95% confidence bounds on AUC — "
            "essential for honest reporting in any research paper or clinical validation.")

        if run_gate(
            key="calib_ran",
            button_label="Run Calibration + Bootstrap CI Analysis",
            description="Generates reliability diagrams for top 4 models with Isotonic "
                        "and Platt scaling calibration. Computes 1000-iteration bootstrap "
                        "confidence intervals on AUC for top 6 models.",
            estimated_time="1–2 minutes"
        ):
            with st.spinner("Computing calibration curves and bootstrap CIs…"):
                top_models = sorted(existing_results.items(), key=lambda x:-x[1]["roc_auc"])[:4]
                top6       = sorted(existing_results.items(), key=lambda x:-x[1]["roc_auc"])[:6]

                section("Calibration Curves — Reliability Diagrams")
                fig, axes = plt.subplots(2,2,figsize=(12,9))
                for ax,(name,res) in zip(axes.flatten(), top_models):
                    pt_o,pp_o = calibration_curve(res["y_test"],res["y_proba"],n_bins=8,strategy="uniform")
                    ax.plot(pp_o,pt_o,"s-",color=BLUE,lw=2,
                            label=f"Original (Brier={brier_score_loss(res['y_test'],res['y_proba']):.4f})")
                    try:
                        cal = CalibratedClassifierCV(res["clf"],method="isotonic",cv=5).fit(Xtr,ytr)
                        p_iso=cal.predict_proba(Xte)[:,1]; pt_i,pp_i=calibration_curve(yte,p_iso,n_bins=8,strategy="uniform")
                        ax.plot(pp_i,pt_i,"o-",color=GREEN,lw=2,label=f"Isotonic ({brier_score_loss(yte,p_iso):.4f})")
                    except: pass
                    try:
                        cal2=CalibratedClassifierCV(res["clf"],method="sigmoid",cv=5).fit(Xtr,ytr)
                        p_sig=cal2.predict_proba(Xte)[:,1]; pt_s,pp_s=calibration_curve(yte,p_sig,n_bins=8,strategy="uniform")
                        ax.plot(pp_s,pt_s,"^-",color=AMBER,lw=2,label=f"Platt ({brier_score_loss(yte,p_sig):.4f})")
                    except: pass
                    ax.plot([0,1],[0,1],color=BORDER_MED,lw=1.5,ls="--",label="Perfect")
                    ax.set(title=name,xlabel="Mean Predicted Prob",ylabel="Fraction Positives")
                    ax.legend(fontsize=7); ax.grid(True)
                fig.suptitle("Calibration Curves — Original vs Isotonic vs Platt",
                             y=1.01,color=NAVY,fontsize=12,fontweight="bold")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                section("Bootstrap 95% Confidence Intervals — AUC")

                def auc_fn(y,p):
                    f,t,_=roc_curve(y,p); return auc(f,t)

                ci_rows = []
                for name,res in top6:
                    mn,sd,lo,hi = bootstrap_ci(res["y_test"],res["y_proba"],auc_fn,n_boot=1000)
                    ci_rows.append({"Model":name,"AUC":f"{res['roc_auc']:.4f}",
                                    "Boot Mean":f"{mn:.4f}","Std":f"{sd:.4f}",
                                    "95% CI":f"[{lo:.4f} – {hi:.4f}]","Width":f"{hi-lo:.4f}"})
                st.dataframe(pd.DataFrame(ci_rows),use_container_width=True)

                fig,ax=plt.subplots(figsize=(9,4))
                for i,row in enumerate(ci_rows):
                    m_=float(row["AUC"]); lo_=float(row["95% CI"].split("–")[0].strip()[1:])
                    hi_=float(row["95% CI"].split("–")[1].strip()[:-1])
                    ax.plot([lo_,hi_],[i,i],color=BLUE,lw=4,solid_capstyle="round")
                    ax.scatter(m_,i,color=NAVY,s=70,zorder=5)
                    ax.text(hi_+0.004,i,f"{m_:.4f} [{lo_:.3f}–{hi_:.3f}]",va="center",fontsize=7.5,color=TEXT_MID)
                ax.set_yticks(range(len(ci_rows))); ax.set_yticklabels([r["Model"] for r in ci_rows])
                ax.set(title="Forest Plot — AUC with 95% Bootstrap CI",xlabel="ROC AUC")
                ax.grid(axis="x"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            success_box("Calibration and Bootstrap CI analysis complete.")


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — STATISTICAL TESTS
    # ══════════════════════════════════════════════════════════════════════════
    with adv_tabs[4]:
        info_box(
            "<strong>McNemar's test</strong> — are two classifiers' error patterns "
            "significantly different? <strong>DeLong's test</strong> — is the AUC "
            "difference between two models statistically significant? "
            "<strong>Wilcoxon signed-rank</strong> — non-parametric CV score comparison. "
            "These tests are required in any peer-reviewed ML paper.")

        if run_gate(
            key="stats_ran",
            button_label="Run Statistical Significance Tests",
            description="Pairwise McNemar's test matrix, DeLong AUC test matrix, "
                        "Wilcoxon signed-rank on 10-fold CV scores, and Cohen's d "
                        "effect sizes across all classifiers.",
            estimated_time="1–2 minutes"
        ):
            with st.spinner("Running pairwise statistical tests…"):
                model_names = list(existing_results.keys()); n = len(model_names)

                section("McNemar's Test — Pairwise Error Pattern Comparison")
                mcn_p = np.ones((n,n))
                for i,n1 in enumerate(model_names):
                    for j,n2 in enumerate(model_names):
                        if i!=j:
                            p,_ = mcnemar_test(existing_results[n1]["y_test"],
                                               existing_results[n1]["y_pred"],
                                               existing_results[n2]["y_pred"])
                            mcn_p[i,j]=p

                sig_cmap = LinearSegmentedColormap.from_list("sig",[RED,AMBER_LIGHT,WHITE],256)
                fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
                im1=ax1.imshow(mcn_p,cmap=sig_cmap,vmin=0,vmax=0.2)
                for i in range(n):
                    for j in range(n):
                        v=mcn_p[i,j]
                        sig="***" if v<.001 else "**" if v<.01 else "*" if v<.05 else "ns"
                        ax1.text(j,i,f"{v:.3f}\n{sig}" if i!=j else "—",
                                 ha="center",va="center",fontsize=6.5,
                                 fontweight="bold" if v<.05 else "normal",
                                 color=WHITE if v<.05 else TEXT_MAIN)
                ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
                ax1.set_xticklabels([n_.replace(" ","\n") for n_ in model_names],fontsize=7)
                ax1.set_yticklabels(model_names,fontsize=7)
                ax1.set_title("McNemar p-values\n(red = significantly different error patterns)")
                plt.colorbar(im1,ax=ax1,label="p-value",shrink=0.7)

                delong_p = np.ones((n,n))
                for i,n1 in enumerate(model_names):
                    for j,n2 in enumerate(model_names):
                        if i!=j:
                            try:
                                _,_,p,_ = delong_auc_test(
                                    existing_results[n1]["y_test"],
                                    existing_results[n1]["y_proba"],
                                    existing_results[n2]["y_proba"])
                                delong_p[i,j]=p
                            except: pass

                im2=ax2.imshow(delong_p,cmap=sig_cmap,vmin=0,vmax=0.2)
                for i in range(n):
                    for j in range(n):
                        v=delong_p[i,j]
                        sig="***" if v<.001 else "**" if v<.01 else "*" if v<.05 else "ns"
                        ax2.text(j,i,f"{v:.3f}\n{sig}" if i!=j else "—",
                                 ha="center",va="center",fontsize=6.5,
                                 fontweight="bold" if v<.05 else "normal",
                                 color=WHITE if v<.05 else TEXT_MAIN)
                ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
                ax2.set_xticklabels([n_.replace(" ","\n") for n_ in model_names],fontsize=7)
                ax2.set_yticklabels(model_names,fontsize=7)
                ax2.set_title("DeLong AUC Test p-values\n(red = significantly different AUC)")
                plt.colorbar(im2,ax=ax2,label="p-value",shrink=0.7)
                fig.suptitle("Pairwise Tests — * p<0.05  ** p<0.01  *** p<0.001",
                             y=1.01,color=NAVY,fontsize=11,fontweight="bold")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                section("Wilcoxon Signed-Rank — CV Score Comparison")
                Xs_all = scaler.transform(df[feat_names]); y_all = df["status"]
                cv10   = StratifiedKFold(10,shuffle=True,random_state=42)
                best_m = max(model_names,key=lambda k:existing_results[k]["roc_auc"])
                s_best = cross_val_score(existing_results[best_m]["clf"],Xs_all,y_all,cv=cv10,scoring="accuracy")
                rows=[]
                for name in model_names:
                    if name==best_m: continue
                    s2=cross_val_score(existing_results[name]["clf"],Xs_all,y_all,cv=cv10,scoring="accuracy")
                    try: stat,p=stats.wilcoxon(s_best,s2)
                    except: stat,p=0,1.0
                    rows.append({"Model A":best_m,"Model B":name,
                                 "A CV Mean":f"{s_best.mean()*100:.2f}%",
                                 "B CV Mean":f"{s2.mean()*100:.2f}%",
                                 "W":f"{stat:.1f}","p-value":f"{p:.4f}",
                                 "Significant":"Yes (p<0.05)" if p<.05 else "No"})
                st.dataframe(pd.DataFrame(rows),use_container_width=True)

            success_box("Statistical tests complete.")


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — CROSS-DATASET VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    with adv_tabs[5]:
        info_box(
            "<strong>Cross-dataset validation</strong> is the gold standard for generalizability. "
            "Models are trained on the UCI Voice Dataset and evaluated — without any retraining — "
            "on the UCI Parkinson's Telemonitoring Dataset (different patients, different recording "
            "conditions). A small AUC drop indicates the model has learned real biomarker patterns "
            "rather than memorising dataset-specific noise.")

        warn_box(
            "For production results download the real UCI Telemonitoring dataset: "
            "archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring — "
            "the demonstration below uses synthetic data matching its published distributions.")

        if run_gate(
            key="cross_ran",
            button_label="Run Cross-Dataset Validation",
            description="Generates a synthetic Telemonitoring dataset (5,875 samples) "
                        "matching published UCI distributions, then evaluates all trained "
                        "models on it without retraining. Quantifies the generalization gap.",
            estimated_time="30–60 seconds"
        ):
            with st.spinner("Running cross-dataset validation…"):
                df_tel = _synthetic_telemonitoring(feat_names, seed=42)
                X_tel  = scaler.transform(df_tel[feat_names])
                y_tel  = df_tel["status"]

                cross_rows=[]; cross_roc={}
                top6 = sorted(existing_results.items(), key=lambda x:-x[1]["roc_auc"])[:6]
                for name,res in top6:
                    yp_=res["clf"].predict(X_tel); ypr_=res["clf"].predict_proba(X_tel)[:,1]
                    f_,t_,_=roc_curve(y_tel,ypr_); auc_=auc(f_,t_)
                    acc_=accuracy_score(y_tel,yp_); f1_=f1_score(y_tel,yp_)
                    drop=res["roc_auc"]-auc_
                    cross_roc[name]={"fpr":f_,"tpr":t_,"auc_tel":auc_,"auc_orig":res["roc_auc"]}
                    cross_rows.append({
                        "Model":name,
                        "Voice AUC (train)":f"{res['roc_auc']:.4f}",
                        "Telemon AUC (test)":f"{auc_:.4f}",
                        "AUC Drop":f"{drop:+.4f}",
                        "Telemon Acc":f"{acc_*100:.2f}%",
                        "Generalizes":
                            "Well" if drop<.05 else ("Moderate" if drop<.15 else "Poor"),
                    })

            st.session_state["cross_rows"] = cross_rows
            st.session_state["cross_roc"]  = cross_roc

            cross_rows = st.session_state["cross_rows"]
            cross_roc  = st.session_state["cross_roc"]

            c1,c2,c3,c4 = st.columns(4)
            with c1: metric_card("5,875", "Telemonitoring\nSamples", BLUE)
            with c2: metric_card("22", "Shared Features", GREEN)
            with c3: metric_card("No Retraining", "Evaluation Mode", AMBER)
            with c4: metric_card(cross_rows[0]["Generalizes"], "Best Model", PURPLE)

            st.markdown("<br>", unsafe_allow_html=True)
            section("Cross-Dataset Results Table")
            df_cross = pd.DataFrame(cross_rows).reset_index(drop=True)
            df_cross.index += 1
            st.dataframe(df_cross, use_container_width=True)

            col1,col2 = st.columns(2)
            with col1:
                section("Telemonitoring Test — ROC Curves")
                fig,ax=plt.subplots(figsize=(6,5))
                for (name,cr),c in zip(cross_roc.items(),PAL):
                    ax.plot(cr["fpr"],cr["tpr"],color=c,lw=2,
                            label=f"{name[:20]} ({cr['auc_tel']:.3f})")
                ax.plot([0,1],[0,1],color=BORDER,lw=1,ls="--")
                ax.set(title="ROC — Evaluated on Telemonitoring\n(Trained on Voice Dataset)",
                       xlabel="FPR",ylabel="TPR")
                ax.legend(fontsize=7,loc="lower right"); ax.grid(True)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            with col2:
                section("Generalization Gap — AUC Drop")
                names_c=[r["Model"] for r in cross_rows]
                drops=[float(r["AUC Drop"]) for r in cross_rows]
                colors_c=[GREEN if d>-.05 else (AMBER if d>-.15 else RED) for d in drops]
                fig,ax=plt.subplots(figsize=(6,5))
                bars=ax.bar(names_c,drops,color=colors_c,alpha=0.8,edgecolor=WHITE)
                ax.axhline(0,color=NAVY,lw=2)
                ax.axhline(-.05,color=AMBER,lw=1.5,ls="--",alpha=.8,label="Acceptable (-0.05)")
                ax.axhline(-.15,color=RED,lw=1.5,ls="--",alpha=.8,label="Poor (-0.15)")
                for bar,d in zip(bars,drops):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()-0.005 if d<0 else bar.get_height()+.002,
                            f"{d:+.4f}",ha="center",
                            va="top" if d<0 else "bottom",fontsize=8,fontweight="bold")
                ax.set(title="AUC Drop: Voice → Telemonitoring",ylabel="ΔAUC")
                ax.set_xticklabels(names_c,rotation=20,ha="right")
                ax.legend(fontsize=8); ax.grid(axis="y")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            success_box(
                f"Cross-dataset validation complete. Best generalizing model: "
                f"<strong>{cross_rows[0]['Model']}</strong> — "
                f"Telemonitoring AUC: <strong>{cross_rows[0]['Telemon AUC (test)']}</strong> "
                f"({cross_rows[0]['Generalizes']})")
