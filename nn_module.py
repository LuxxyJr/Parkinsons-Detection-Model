"""
Neural Network Module — Parkinson's Detection Lab
All heavy training gated behind run buttons.
Streamlit Cloud safe — nothing trains on page load.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_curve, auc, confusion_matrix,
    classification_report, precision_score, recall_score,
    matthews_corrcoef, average_precision_score, precision_recall_curve
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── Colours ──────────────────────────────────────────────────────────────────
WHITE = "#ffffff"; BG = "#f4f6f9"; NAVY = "#1e3a5f"; BLUE = "#2563eb"
BLUE_LIGHT = "#dbeafe"; GREEN = "#059669"; GREEN_LIGHT = "#d1fae5"
RED = "#dc2626"; AMBER = "#d97706"; PURPLE = "#7c3aed"; SLATE = "#0ea5e9"
BORDER = "#e2e8f0"; BORDER_MED = "#cbd5e1"
TEXT_MAIN = "#1e293b"; TEXT_MID = "#475569"; TEXT_DIM = "#94a3b8"
PAL = [BLUE, GREEN, AMBER, PURPLE, RED, SLATE]

plt.rcParams.update({
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "axes.edgecolor": BORDER_MED, "axes.titlecolor": NAVY,
    "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": BORDER,
    "grid.linestyle": "--", "grid.alpha": 0.8,
    "legend.facecolor": WHITE, "legend.edgecolor": BORDER,
    "xtick.color": TEXT_DIM, "ytick.color": TEXT_DIM,
    "text.color": TEXT_MAIN,
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
        f'<div style="background:#fef3c7;border-left:4px solid {AMBER};'
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
        f'<div style="font-family:DM Serif Display,serif;font-size:1.8rem;'
        f'color:{color};line-height:1.1">{val}</div>'
        f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:1.5px;'
        f'text-transform:uppercase;color:{TEXT_DIM};margin-top:4px">{label}</div>'
        f'</div>', unsafe_allow_html=True)

def run_gate(key, button_label, description, estimated_time, warning=None):
    """Returns True if computation has been triggered, False otherwise."""
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
            f'<div style="font-size:0.84rem;color:{TEXT_MID};line-height:1.7;'
            f'margin-bottom:16px">{description}</div>'
            f'<div style="display:inline-block;background:#fef3c7;border:1px solid #fde68a;'
            f'border-radius:20px;padding:4px 16px;font-size:0.72rem;font-weight:600;'
            f'color:#92400e;margin-bottom:20px">Estimated time: {estimated_time}</div>'
            f'</div>', unsafe_allow_html=True)
        if warning:
            warn_box(warning)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"Run {button_label}", key=f"btn_{key}"):
            st.session_state[key] = True
            st.rerun()
        return False
    return True


def get_metrics(clf, Xte, yte, name=""):
    yp = clf.predict(Xte)
    ypr = clf.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, ypr)
    pr_p, pr_r, _ = precision_recall_curve(yte, ypr)
    return dict(
        name=name, accuracy=accuracy_score(yte, yp),
        f1=f1_score(yte, yp), roc_auc=auc(fpr, tpr),
        pr_auc=average_precision_score(yte, ypr),
        mcc=matthews_corrcoef(yte, yp),
        precision=precision_score(yte, yp),
        recall=recall_score(yte, yp),
        cm=confusion_matrix(yte, yp),
        report=classification_report(yte, yp, target_names=["Healthy", "PD"]),
        fpr=fpr, tpr=tpr, pr_p=pr_p, pr_r=pr_r,
        y_pred=yp, y_proba=ypr, y_test=yte,
    )


# ── PyTorch model ─────────────────────────────────────────────────────────────
if TORCH_OK:
    class PDNet(nn.Module):
        def __init__(self, input_dim, hidden_layers, dropout=0.3, batchnorm=True):
            super().__init__()
            layers = []; in_d = input_dim
            for i, h in enumerate(hidden_layers):
                layers.append(nn.Linear(in_d, h))
                if batchnorm: layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                if dropout > 0 and i < len(hidden_layers) - 1:
                    layers.append(nn.Dropout(dropout))
                in_d = h
            layers += [nn.Linear(in_d, 1), nn.Sigmoid()]
            self.net = nn.Sequential(*layers)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        def forward(self, x): return self.net(x).squeeze(1)
        def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

    class EarlyStopping:
        def __init__(self, patience=20, min_delta=0.001):
            self.patience = patience; self.min_delta = min_delta
            self.best = np.inf; self.counter = 0; self.best_state = None

        def __call__(self, val_loss, model):
            if val_loss < self.best - self.min_delta:
                self.best = val_loss; self.counter = 0
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                self.counter += 1
            return self.counter >= self.patience

    def train_pdnet(Xtr, ytr, Xva, yva, config, n_epochs=200, seed=42):
        torch.manual_seed(seed)
        Xt = torch.FloatTensor(Xtr)
        yt = torch.FloatTensor(ytr.values if hasattr(ytr, "values") else ytr)
        Xv = torch.FloatTensor(Xva)
        yv = torch.FloatTensor(yva.values if hasattr(yva, "values") else yva)

        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=config.get("batch_size", 32), shuffle=True)
        model = PDNet(Xtr.shape[1], config["hidden_layers"],
                      config.get("dropout", 0.3), config.get("batchnorm", True))
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001),
                               weight_decay=config.get("wd", 1e-4))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=8, factor=0.5, min_lr=1e-6)
        es = EarlyStopping(patience=20)

        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
                "lr": [], "stopped_epoch": n_epochs}

        for epoch in range(n_epochs):
            model.train(); epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            model.eval()
            with torch.no_grad():
                vl = criterion(model(Xv), yv).item()
                va = ((model(Xv) >= 0.5).float() == yv).float().mean().item()
                tl = epoch_loss / len(Xtr)
                ta = ((model(Xt) >= 0.5).float() == yt).float().mean().item()

            scheduler.step(vl)
            hist["train_loss"].append(tl); hist["val_loss"].append(vl)
            hist["train_acc"].append(ta);  hist["val_acc"].append(va)
            hist["lr"].append(optimizer.param_groups[0]["lr"])

            if es(vl, model):
                hist["stopped_epoch"] = epoch + 1
                model.load_state_dict(es.best_state)
                break

        return model, hist

    def eval_pdnet(model, X, y_true):
        model.eval()
        with torch.no_grad():
            proba = model(torch.FloatTensor(X)).numpy()
        pred = (proba >= 0.5).astype(int)
        y = y_true.values if hasattr(y_true, "values") else np.array(y_true)
        fpr, tpr, _ = roc_curve(y, proba)
        pr_p, pr_r, _ = precision_recall_curve(y, proba)
        return dict(
            proba=proba, pred=pred,
            accuracy=accuracy_score(y, pred), f1=f1_score(y, pred),
            roc_auc=auc(fpr, tpr), pr_auc=average_precision_score(y, proba),
            mcc=matthews_corrcoef(y, pred),
            cm=confusion_matrix(y, pred),
            report=classification_report(y, pred, target_names=["Healthy", "PD"]),
            fpr=fpr, tpr=tpr, pr_p=pr_p, pr_r=pr_r,
            y_pred=pred, y_proba=proba, y_test=y,
        )

ARCH_CONFIGS = {
    "Shallow (64)":              {"hidden_layers": [64],           "dropout": 0.2, "batchnorm": True,  "lr": 0.001,  "batch_size": 32, "wd": 1e-4},
    "Medium (128-64)":           {"hidden_layers": [128, 64],      "dropout": 0.3, "batchnorm": True,  "lr": 0.001,  "batch_size": 32, "wd": 1e-4},
    "Deep (256-128-64)":         {"hidden_layers": [256, 128, 64], "dropout": 0.3, "batchnorm": True,  "lr": 0.001,  "batch_size": 16, "wd": 1e-4},
    "Wide (512-256)":            {"hidden_layers": [512, 256],     "dropout": 0.4, "batchnorm": True,  "lr": 0.0005, "batch_size": 32, "wd": 1e-3},
    "Deep-Narrow (256-128-64-32)":{"hidden_layers": [256,128,64,32],"dropout": 0.35,"batchnorm": True, "lr": 0.001,  "batch_size": 16, "wd": 1e-4},
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════
def render_nn_tab(df, scaler, feat_names, existing_results):
    df_hash = len(df)
    X = df[feat_names]; y = df["status"]
    Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=42)

    nn_tabs = st.tabs([
        "MLP Architecture Search",
        "PyTorch Deep NN",
        "NN vs Classical ML",
        "Architecture Explorer",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    #  SUB-TAB 0 — MLP SEARCH
    # ══════════════════════════════════════════════════════════════════════════
    with nn_tabs[0]:
        info_box(
            "<strong>MLP Architecture Search</strong> via GridSearchCV over "
            "42 configurations — 7 layer configurations × 2 activation functions "
            "(ReLU, tanh) × 3 L2 regularisation strengths — with 5-fold stratified CV "
            "and built-in early stopping. Finds the optimal shallow network for this dataset.")

        if run_gate(
            key="mlp_ran",
            button_label="MLP Architecture Search",
            description="GridSearchCV across 42 MLP configurations with 5-fold CV. "
                        "Optimises ROC AUC. Includes early stopping and adaptive learning rate.",
            estimated_time="3–5 minutes on Streamlit Cloud"
        ):
            with st.spinner("Running MLP GridSearchCV (42 configs × 5-fold CV)…"):
                param_grid = {
                    "hidden_layer_sizes": [(64,),(128,),(64,32),(128,64),(128,64,32),(256,128,64),(64,64,64)],
                    "activation":         ["relu","tanh"],
                    "alpha":              [0.0001, 0.001, 0.01],
                    "learning_rate":      ["adaptive"],
                    "max_iter":           [1000],
                    "early_stopping":     [True],
                    "validation_fraction":[0.15],
                }
                search = GridSearchCV(
                    MLPClassifier(random_state=42, solver="adam"),
                    param_grid,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring="roc_auc", n_jobs=-1,
                    return_train_score=True, refit=True,
                )
                search.fit(Xtr, ytr)
                best_mlp = search.best_estimator_
                cv10 = cross_val_score(best_mlp, Xs, y,
                                       cv=StratifiedKFold(10, shuffle=True, random_state=42),
                                       scoring="accuracy")
                m = get_metrics(best_mlp, Xte, yte, "MLP")
                st.session_state["mlp_model"]   = best_mlp
                st.session_state["mlp_metrics"] = m
                st.session_state["mlp_search"]  = search
                st.session_state["mlp_cv10"]    = cv10

            best_mlp = st.session_state["mlp_model"]
            m        = st.session_state["mlp_metrics"]
            search   = st.session_state["mlp_search"]
            cv10     = st.session_state["mlp_cv10"]
            bp       = search.best_params_

            # Best config banner
            st.markdown(
                f'<div style="background:{NAVY};color:{WHITE};border-radius:8px;'
                f'padding:14px 20px;font-size:0.85rem;margin-bottom:20px">'
                f'<strong>Best Architecture Found:</strong> &nbsp;'
                f'Layers: <code style="background:rgba(255,255,255,0.15);padding:2px 8px;'
                f'border-radius:4px">{bp["hidden_layer_sizes"]}</code> &nbsp;'
                f'Activation: <code style="background:rgba(255,255,255,0.15);padding:2px 8px;'
                f'border-radius:4px">{bp["activation"]}</code> &nbsp;'
                f'Alpha: <code style="background:rgba(255,255,255,0.15);padding:2px 8px;'
                f'border-radius:4px">{bp["alpha"]}</code>'
                f'</div>', unsafe_allow_html=True)

            c1,c2,c3,c4 = st.columns(4)
            with c1: metric_card(f"{m['roc_auc']:.4f}", "Test AUC", BLUE)
            with c2: metric_card(f"{m['accuracy']*100:.1f}%", "Test Accuracy", GREEN)
            with c3: metric_card(f"{m['f1']:.4f}", "F1 Score", AMBER)
            with c4: metric_card(f"{search.best_score_:.4f}", "Best CV AUC", PURPLE)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                section("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(5, 4))
                cm_n = m["cm"].astype(float) / m["cm"].sum(axis=1, keepdims=True)
                im = ax.imshow(cm_n, cmap=LinearSegmentedColormap.from_list("c",[WHITE,BLUE_LIGHT,BLUE],256), vmin=0, vmax=1)
                for i in range(2):
                    for j in range(2):
                        ax.text(j,i,f"{m['cm'][i,j]}\n({cm_n[i,j]*100:.1f}%)",
                                ha="center",va="center",fontsize=13,fontweight="bold",
                                color=WHITE if cm_n[i,j]>0.6 else NAVY)
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(["Healthy","PD"]); ax.set_yticklabels(["Healthy","PD"])
                ax.set_title("MLP Best Architecture — Confusion Matrix")
                plt.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            with col2:
                section("Architecture Search Heatmap")
                try:
                    import seaborn as sns
                    cv_r = pd.DataFrame(search.cv_results_)
                    piv = (cv_r.groupby(["param_hidden_layer_sizes","param_activation"])
                           ["mean_test_score"].max().reset_index()
                           .pivot(index="param_hidden_layer_sizes",
                                  columns="param_activation",
                                  values="mean_test_score"))
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(piv, annot=True, fmt=".4f",
                                cmap=LinearSegmentedColormap.from_list("h",[WHITE,BLUE_LIGHT,BLUE],256),
                                ax=ax, linewidths=0.5, linecolor=BORDER,
                                cbar_kws={"shrink":0.7,"label":"Mean CV AUC"},
                                annot_kws={"size":8,"weight":"bold"})
                    ax.set_title("Architecture × Activation — CV AUC")
                    plt.xticks(rotation=0); plt.yticks(rotation=0, fontsize=7)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                except Exception as e:
                    st.info(f"Heatmap skipped: {e}")

            section("Classification Report")
            st.code(m["report"], language="")


    # ══════════════════════════════════════════════════════════════════════════
    #  SUB-TAB 1 — PYTORCH
    # ══════════════════════════════════════════════════════════════════════════
    with nn_tabs[1]:
        if not TORCH_OK:
            warn_box("PyTorch not installed. Run: <code>pip install torch</code>")
        else:
            info_box(
                "<strong>PyTorch architecture search</strong> trains 5 custom deep networks "
                "— from a shallow 64-neuron net to a 4-layer deep-narrow network — each with "
                "BatchNorm, Dropout, Kaiming initialisation, ReduceLROnPlateau scheduling, "
                "and Early Stopping (patience=20). Best architecture selected by test AUC.")

            if run_gate(
                key="pytorch_ran",
                button_label="PyTorch Architecture Search",
                description="Trains 5 architectures for up to 200 epochs each with early stopping. "
                            "All training curves, leaderboard, and confusion matrices shown after.",
                estimated_time="3–6 minutes on Streamlit Cloud (CPU only)",
                warning="PyTorch runs on CPU on Streamlit Cloud — no GPU available. "
                        "Training is fast enough for this dataset size (195 samples)."
            ):
                arch_results = {}
                progress = st.progress(0, text="Training architectures…")
                for i, (arch_name, config) in enumerate(ARCH_CONFIGS.items()):
                    progress.progress((i) / len(ARCH_CONFIGS),
                                      text=f"Training {arch_name}…")
                    model, hist = train_pdnet(Xtr2, ytr2, Xva, yva, config,
                                             n_epochs=200, seed=42)
                    metrics = eval_pdnet(model, Xte, yte)
                    arch_results[arch_name] = {
                        "model": model, "history": hist,
                        "config": config, **metrics,
                        "n_params": model.n_params(),
                    }
                progress.progress(1.0, text="Done.")
                best_arch = max(arch_results, key=lambda k: arch_results[k]["roc_auc"])
                st.session_state["pt_results"] = arch_results
                st.session_state["pt_best"]    = best_arch

            if "pt_results" in st.session_state:
                arch_results = st.session_state["pt_results"]
                best_arch    = st.session_state["pt_best"]
                best         = arch_results[best_arch]

                st.markdown(
                    f'<div style="background:{NAVY};color:{WHITE};border-radius:8px;'
                    f'padding:14px 20px;font-size:0.85rem;margin-bottom:20px">'
                    f'<strong>Best Architecture: {best_arch}</strong> &nbsp;|&nbsp; '
                    f'Layers: <code style="background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px">'
                    f'{best["config"]["hidden_layers"]}</code> &nbsp;'
                    f'Dropout: <code style="background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px">'
                    f'{best["config"]["dropout"]}</code> &nbsp;'
                    f'Params: <code style="background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px">'
                    f'{best["n_params"]:,}</code> &nbsp;'
                    f'Stopped @ epoch: <code style="background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px">'
                    f'{best["history"]["stopped_epoch"]}</code>'
                    f'</div>', unsafe_allow_html=True)

                c1,c2,c3,c4 = st.columns(4)
                with c1: metric_card(f"{best['roc_auc']:.4f}", "Test AUC", BLUE)
                with c2: metric_card(f"{best['accuracy']*100:.1f}%", "Test Accuracy", GREEN)
                with c3: metric_card(f"{best['f1']:.4f}", "F1 Score", AMBER)
                with c4: metric_card(f"{best['n_params']:,}", "Parameters", PURPLE)

                st.markdown("<br>", unsafe_allow_html=True)
                section(f"Training Curves — {best_arch}")
                hist = best["history"]
                ep   = range(1, len(hist["train_loss"]) + 1)
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                axes[0].plot(ep, hist["train_loss"], color=BLUE, lw=2, label="Train Loss")
                axes[0].plot(ep, hist["val_loss"],   color=RED,  lw=2, label="Val Loss")
                if hist["stopped_epoch"] < 200:
                    axes[0].axvline(hist["stopped_epoch"], color=AMBER, lw=1.5, ls=":",
                                    label=f"Early stop @{hist['stopped_epoch']}")
                axes[0].set(title="Loss Curves", xlabel="Epoch", ylabel="BCE Loss")
                axes[0].legend(); axes[0].grid(True)

                axes[1].plot(ep, [a*100 for a in hist["train_acc"]], color=BLUE, lw=2, label="Train")
                axes[1].plot(ep, [a*100 for a in hist["val_acc"]],   color=GREEN,lw=2, label="Val")
                if hist["stopped_epoch"] < 200:
                    axes[1].axvline(hist["stopped_epoch"], color=AMBER, lw=1.5, ls=":")
                axes[1].set(title="Accuracy Curves", xlabel="Epoch", ylabel="Accuracy (%)")
                axes[1].legend(); axes[1].grid(True)

                axes[2].plot(ep, hist["lr"], color=PURPLE, lw=2)
                axes[2].set(title="Learning Rate Schedule\n(ReduceLROnPlateau)",
                            xlabel="Epoch", ylabel="LR")
                axes[2].set_yscale("log"); axes[2].grid(True)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                section("All Architectures — Validation Loss Comparison")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
                for (arch_name, res), c in zip(arch_results.items(), PAL):
                    h  = res["history"]
                    ep2 = range(1, len(h["val_loss"]) + 1)
                    lw = 2.5 if arch_name == best_arch else 1.2
                    al = 1.0 if arch_name == best_arch else 0.5
                    tag = "BEST" if arch_name == best_arch else f"AUC={res['roc_auc']:.3f}"
                    ax1.plot(ep2, h["val_loss"],        color=c, lw=lw, alpha=al, label=f"{arch_name} ({tag})")
                    ax2.plot(ep2, [a*100 for a in h["val_acc"]], color=c, lw=lw, alpha=al, label=f"{arch_name} ({tag})")
                ax1.set(title="Validation Loss", xlabel="Epoch", ylabel="Val Loss")
                ax1.legend(fontsize=7); ax1.grid(True)
                ax2.set(title="Validation Accuracy", xlabel="Epoch", ylabel="Val Acc (%)")
                ax2.legend(fontsize=7); ax2.grid(True)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                section("Architecture Leaderboard")
                lb = []
                for arch_name, res in arch_results.items():
                    lb.append({
                        "Architecture": arch_name,
                        "Layers":       str(res["config"]["hidden_layers"]),
                        "Params":       f"{res['n_params']:,}",
                        "Dropout":      res["config"]["dropout"],
                        "Stopped@":     res["history"]["stopped_epoch"],
                        "Val Loss":     f"{min(res['history']['val_loss']):.4f}",
                        "Test AUC":     f"{res['roc_auc']:.4f}",
                        "Test Acc":     f"{res['accuracy']*100:.2f}%",
                        "F1":           f"{res['f1']:.4f}",
                        "Best":         "*** BEST ***" if arch_name == best_arch else "",
                    })
                lb_df = pd.DataFrame(lb).sort_values("Test AUC", ascending=False).reset_index(drop=True)
                lb_df.index += 1
                st.dataframe(lb_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    section("Best Model — Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    cm_n = best["cm"].astype(float) / best["cm"].sum(axis=1, keepdims=True)
                    im = ax.imshow(cm_n, cmap=LinearSegmentedColormap.from_list("c",[WHITE,BLUE_LIGHT,BLUE],256), vmin=0, vmax=1)
                    for i in range(2):
                        for j in range(2):
                            ax.text(j,i,f"{best['cm'][i,j]}\n({cm_n[i,j]*100:.1f}%)",
                                    ha="center",va="center",fontsize=13,fontweight="bold",
                                    color=WHITE if cm_n[i,j]>0.6 else NAVY)
                    ax.set_xticks([0,1]); ax.set_yticks([0,1])
                    ax.set_xticklabels(["Healthy","PD"]); ax.set_yticklabels(["Healthy","PD"])
                    ax.set_title(f"PyTorch ({best_arch})")
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                with col2:
                    section("Best Model — ROC Curve")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(best["fpr"], best["tpr"], color=BLUE, lw=2.5,
                            label=f"PyTorch AUC = {best['roc_auc']:.4f}")
                    ax.fill_between(best["fpr"], best["tpr"], alpha=0.08, color=BLUE)
                    ax.plot([0,1],[0,1], color=BORDER_MED, lw=1, ls="--")
                    ax.set(title=f"PyTorch ({best_arch}) — ROC", xlabel="FPR", ylabel="TPR")
                    ax.legend(); ax.grid(True)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


    # ══════════════════════════════════════════════════════════════════════════
    #  SUB-TAB 2 — NN VS CLASSICAL
    # ══════════════════════════════════════════════════════════════════════════
    with nn_tabs[2]:
        info_box(
            "Head-to-head comparison between all classical ML models and "
            "the trained neural networks. Run the MLP and PyTorch tabs first "
            "to include them in this comparison.")

        all_models = {}
        for name, res in existing_results.items():
            all_models[name] = {**res, "type": "Classical"}
        if "mlp_metrics" in st.session_state:
            all_models["MLP (Best Arch)"] = {**st.session_state["mlp_metrics"], "type": "Neural Network"}
        if "pt_results" in st.session_state:
            best_pt = st.session_state["pt_results"][st.session_state["pt_best"]]
            all_models[f"PyTorch ({st.session_state['pt_best']})"] = {**best_pt, "type": "Neural Network"}

        if len(all_models) == len(existing_results):
            st.info("Run the MLP and/or PyTorch tabs first to add neural networks to this comparison.")

        rows = []
        for name, m in all_models.items():
            rows.append({
                "Model": name, "Type": m["type"],
                "Accuracy": f"{m['accuracy']*100:.2f}%",
                "F1": f"{m['f1']:.4f}",
                "ROC AUC": f"{m['roc_auc']:.4f}",
                "PR AUC": f"{m['pr_auc']:.4f}",
            })
        st.dataframe(
            pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True),
            use_container_width=True)

        section("ROC Curves — Classical ML vs Neural Networks")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for (name, m), c in zip(all_models.items(), PAL * 3):
            is_nn = m["type"] == "Neural Network"
            ax.plot(m["fpr"], m["tpr"],
                    color=c, lw=2.8 if is_nn else 1.2,
                    ls="-" if is_nn else "--",
                    alpha=1.0 if is_nn else 0.5,
                    label=f"{'★ ' if is_nn else ''}{name} ({m['roc_auc']:.3f})")
        ax.plot([0,1],[0,1], color=BORDER, lw=1, ls=":")
        ax.set(title="ROC Curves — All Models (★ = Neural Networks)", xlabel="FPR", ylabel="TPR")
        ax.legend(loc="lower right", fontsize=7.5); ax.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        section("AUC Comparison — All Models")
        names_ = list(all_models.keys())
        aucs_  = [all_models[n]["roc_auc"] for n in names_]
        colors_= [BLUE if all_models[n]["type"]=="Neural Network" else TEXT_DIM for n in names_]
        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(names_, aucs_, color=colors_, alpha=0.85, edgecolor=WHITE)
        for bar, val in zip(bars, aucs_):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticklabels([n.replace(" ","\n") for n in names_], fontsize=7.5)
        ax.set(title="ROC AUC — Classical ML (grey) vs Neural Networks (blue)",
               ylabel="ROC AUC", ylim=[0.6, 1.05])
        ax.grid(axis="y")
        handles = [mpatches.Patch(color=BLUE, label="Neural Network"),
                   mpatches.Patch(color=TEXT_DIM, label="Classical ML")]
        ax.legend(handles=handles, fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


    # ══════════════════════════════════════════════════════════════════════════
    #  SUB-TAB 3 — ARCHITECTURE EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    with nn_tabs[3]:
        section("Architecture Explorer — Visualise Any Network")
        arch_choice = st.selectbox("Select architecture:", list(ARCH_CONFIGS.keys()))
        config = ARCH_CONFIGS[arch_choice]
        layers = [22] + config["hidden_layers"] + [1]
        total_params = sum(layers[i]*layers[i+1]+layers[i+1] for i in range(len(layers)-1))

        c1,c2,c3 = st.columns(3)
        with c1: metric_card(len(config["hidden_layers"]), "Hidden Layers", BLUE)
        with c2: metric_card(f"{total_params:,}", "Total Parameters", GREEN)
        with c3: metric_card(config["dropout"], "Dropout Rate", AMBER)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Network Diagram")

        layer_names  = (["Input\n(22 features)"] +
                        [f"Hidden {i+1}\n({s} neurons)" for i,s in enumerate(config["hidden_layers"])] +
                        ["Output\n(PD prob)"])
        layer_colors = [SLATE] + [BLUE]*len(config["hidden_layers"]) + [GREEN]

        fig, ax = plt.subplots(figsize=(max(10, len(layers)*2.5), 6))
        ax.set_xlim(-0.5, len(layers)-0.5); ax.set_ylim(-2, 6)
        ax.set_facecolor(BG); ax.axis("off")
        ax.set_title(f"Architecture: {arch_choice}", color=NAVY, fontsize=12, fontweight="bold")

        node_positions = []
        for i, (size, color) in enumerate(zip(layers, layer_colors)):
            display = min(size, 8)
            y_start = -(display-1)/2
            positions = [(i, y_start+j) for j in range(display)]
            node_positions.append(positions)
            if i > 0:
                for (px,py) in node_positions[i-1][:5]:
                    for (cx,cy) in positions[:5]:
                        ax.plot([px+0.15,cx-0.15],[py,cy], color=BORDER_MED, lw=0.4, alpha=0.5, zorder=1)
            for (x,y) in positions:
                ax.add_patch(plt.Circle((x,y), 0.12, color=color, alpha=0.85, zorder=3))
            if size > 8:
                ax.text(i, -(display/2)-0.5, f"...+{size-8}", ha="center", fontsize=7, color=TEXT_DIM)
            ax.text(i, -(display/2)-1.0, layer_names[i], ha="center", fontsize=8,
                    color=NAVY, fontweight="bold")
            annot = ("ReLU + Dropout" if i>0 and i<len(layers)-1 and config["dropout"]>0
                     else ("Input Layer" if i==0 else "Sigmoid Output"))
            if config.get("batchnorm") and 0 < i < len(layers)-1:
                annot += "\nBatchNorm"
            ax.text(i, max(p[1] for p in positions)+0.5, annot,
                    ha="center", fontsize=7, color=TEXT_MID, style="italic")

        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        section("Full Configuration")
        config_display = {
            "Architecture":        arch_choice,
            "Input Dimension":     22,
            "Hidden Layers":       str(config["hidden_layers"]),
            "Output":              "1 neuron → Sigmoid → PD probability",
            "Activation":          "ReLU (hidden layers) + Sigmoid (output)",
            "Batch Normalisation": "Yes" if config.get("batchnorm") else "No",
            "Dropout Rate":        config["dropout"],
            "Optimiser":           "Adam",
            "Learning Rate":       config["lr"],
            "LR Scheduler":        "ReduceLROnPlateau (patience=8, factor=0.5)",
            "Weight Decay (L2)":   config["wd"],
            "Batch Size":          config["batch_size"],
            "Max Epochs":          200,
            "Early Stopping":      "Yes — patience=20, min_delta=0.001",
            "Weight Init":         "Kaiming Normal (He initialisation)",
            "Loss Function":       "Binary Cross-Entropy (BCE)",
            "Estimated Params":    f"{total_params:,}",
        }
        for k,v in config_display.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.82rem;padding:6px 0;border-bottom:1px solid {BORDER}">'
                f'<span style="color:{TEXT_MID};font-weight:500">{k}</span>'
                f'<span style="color:{NAVY};font-weight:600">{v}</span></div>',
                unsafe_allow_html=True)
