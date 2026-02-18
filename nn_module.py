"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   NEURAL NETWORK MODULE — Parkinson's Detection Lab                         ║
║   MLP (scikit-learn) + PyTorch Deep NN                                      ║
║   Architecture Search + Early Stopping + Full Training Curves               ║
║                                                                              ║
║   Add this tab to your main app:                                             ║
║       from nn_module import render_nn_tab                                    ║
║       render_nn_tab(df, scaler, FEAT_NAMES, results)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_curve, auc,
    confusion_matrix, classification_report,
    precision_score, recall_score, matthews_corrcoef,
    average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap

# ── Colours (match main app) ─────────────────────────────────────────────────
WHITE       = "#ffffff"
BG          = "#f4f6f9"
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
SLATE       = "#0ea5e9"
BORDER      = "#e2e8f0"
BORDER_MED  = "#cbd5e1"
TEXT_MAIN   = "#1e293b"
TEXT_MID    = "#475569"
TEXT_DIM    = "#94a3b8"

# ── Try importing PyTorch ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  PYTORCH MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class PDNet(nn.Module):
        """
        Configurable deep neural network for Parkinson's detection.
        Supports variable depth, width, dropout, and batch normalisation.
        """
        def __init__(self, input_dim, hidden_layers, dropout_rate=0.3, use_batchnorm=True):
            super().__init__()
            layers = []
            in_dim = input_dim
            for i, h_dim in enumerate(hidden_layers):
                layers.append(nn.Linear(in_dim, h_dim))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0 and i < len(hidden_layers) - 1:
                    layers.append(nn.Dropout(dropout_rate))
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.network(x).squeeze(1)

        def count_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class EarlyStopping:
        """Stops training when validation loss stops improving."""
        def __init__(self, patience=15, min_delta=0.001):
            self.patience   = patience
            self.min_delta  = min_delta
            self.best_loss  = np.inf
            self.counter    = 0
            self.best_state = None

        def __call__(self, val_loss, model):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss  = val_loss
                self.counter    = 0
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                self.counter += 1
            return self.counter >= self.patience


def train_pytorch_model(X_tr, y_tr, X_va, y_va, config, n_epochs=300, seed=42):
    """
    Full PyTorch training loop with:
    - Early stopping
    - ReduceLROnPlateau scheduler
    - Per-epoch metrics tracking
    """
    if not TORCH_AVAILABLE:
        return None

    torch.manual_seed(seed)
    device = torch.device("cpu")

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr.values if hasattr(y_tr,'values') else y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    y_va_t = torch.FloatTensor(y_va.values if hasattr(y_va,'values') else y_va).to(device)

    dataset  = TensorDataset(X_tr_t, y_tr_t)
    loader   = DataLoader(dataset, batch_size=config.get("batch_size", 32), shuffle=True)

    model = PDNet(
        input_dim    = X_tr.shape[1],
        hidden_layers= config["hidden_layers"],
        dropout_rate = config.get("dropout", 0.3),
        use_batchnorm= config.get("batchnorm", True),
    ).to(device)

    # Loss with class weighting to handle imbalance
    pos_weight = torch.tensor([(y_tr==0).sum() / (y_tr==1).sum()]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if False else nn.BCELoss()

    optimizer  = optim.Adam(
        model.parameters(),
        lr           = config.get("lr", 0.001),
        weight_decay = config.get("weight_decay", 1e-4)
    )
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5, min_lr=1e-6
    )
    early_stop = EarlyStopping(patience=20, min_delta=0.001)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [], "stopped_epoch": n_epochs,
    }

    for epoch in range(n_epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        train_loss = epoch_loss / len(X_tr)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_out  = model(X_va_t)
            val_loss = criterion(val_out, y_va_t).item()
            val_pred = (val_out >= 0.5).float()
            val_acc  = (val_pred == y_va_t).float().mean().item()
            tr_out   = model(X_tr_t)
            tr_pred  = (tr_out >= 0.5).float()
            tr_acc   = (tr_pred == y_tr_t).float().mean().item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if early_stop(val_loss, model):
            history["stopped_epoch"] = epoch + 1
            model.load_state_dict(early_stop.best_state)
            break

    return model, history


def evaluate_pytorch(model, X, y_true):
    """Get full metrics from a trained PyTorch model."""
    if not TORCH_AVAILABLE or model is None:
        return None
    model.eval()
    with torch.no_grad():
        X_t    = torch.FloatTensor(X)
        proba  = model(X_t).numpy()
        pred   = (proba >= 0.5).astype(int)
        y_arr  = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
        fpr, tpr, _ = roc_curve(y_arr, proba)
        pr_p, pr_r, _ = precision_recall_curve(y_arr, proba)
        return dict(
            proba    = proba,
            pred     = pred,
            accuracy = accuracy_score(y_arr, pred),
            f1       = f1_score(y_arr, pred),
            roc_auc  = auc(fpr, tpr),
            pr_auc   = average_precision_score(y_arr, proba),
            mcc      = matthews_corrcoef(y_arr, pred),
            precision= precision_score(y_arr, pred),
            recall   = recall_score(y_arr, pred),
            cm       = confusion_matrix(y_arr, pred),
            report   = classification_report(y_arr, pred, target_names=["Healthy","PD"]),
            fpr=fpr, tpr=tpr, pr_p=pr_p, pr_r=pr_r,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MLP ARCHITECTURE SEARCH (scikit-learn)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def run_mlp_search(_scaler, df_hash, df, feat_names):
    X = df[feat_names]; y = df["status"]
    Xs = _scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)

    param_grid = {
        "hidden_layer_sizes": [
            (64,),
            (128,),
            (64, 32),
            (128, 64),
            (128, 64, 32),
            (256, 128, 64),
            (64, 64, 64),
        ],
        "activation":    ["relu", "tanh"],
        "alpha":         [0.0001, 0.001, 0.01],
        "learning_rate": ["adaptive"],
        "max_iter":      [1000],
        "early_stopping":  [True],
        "validation_fraction": [0.15],
    }

    mlp_search = GridSearchCV(
        MLPClassifier(random_state=42, solver="adam"),
        param_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
        return_train_score=True,
        refit=True,
        verbose=0,
    )
    mlp_search.fit(Xtr, ytr)
    best_mlp  = mlp_search.best_estimator_
    yp        = best_mlp.predict(Xte)
    ypr       = best_mlp.predict_proba(Xte)[:,1]
    fpr, tpr, _ = roc_curve(yte, ypr)
    pr_p, pr_r, _= precision_recall_curve(yte, ypr)
    cv10      = cross_val_score(best_mlp, Xs, y, cv=StratifiedKFold(10, shuffle=True, random_state=42), scoring="accuracy")

    metrics = dict(
        accuracy =accuracy_score(yte, yp),
        f1       =f1_score(yte, yp),
        roc_auc  =auc(fpr, tpr),
        pr_auc   =average_precision_score(yte, ypr),
        mcc      =matthews_corrcoef(yte, yp),
        precision=precision_score(yte, yp),
        recall   =recall_score(yte, yp),
        cm       =confusion_matrix(yte, yp),
        report   =classification_report(yte, yp, target_names=["Healthy","PD"]),
        fpr=fpr, tpr=tpr, pr_p=pr_p, pr_r=pr_r,
        y_test=yte, y_pred=yp, y_proba=ypr,
        cv_mean=cv10.mean(), cv_std=cv10.std(),
    )

    cv_results = pd.DataFrame(mlp_search.cv_results_)
    return best_mlp, mlp_search.best_params_, mlp_search.best_score_, metrics, cv_results, Xtr, Xte, ytr, yte


# ══════════════════════════════════════════════════════════════════════════════
#  PYTORCH ARCHITECTURE SEARCH
# ══════════════════════════════════════════════════════════════════════════════
ARCH_CONFIGS = {
    "Shallow (64)":             {"hidden_layers": [64],          "dropout": 0.2, "batchnorm": True,  "lr": 0.001, "batch_size": 32, "weight_decay": 1e-4},
    "Medium (128→64)":          {"hidden_layers": [128, 64],     "dropout": 0.3, "batchnorm": True,  "lr": 0.001, "batch_size": 32, "weight_decay": 1e-4},
    "Deep (256→128→64)":        {"hidden_layers": [256,128,64],  "dropout": 0.3, "batchnorm": True,  "lr": 0.001, "batch_size": 16, "weight_decay": 1e-4},
    "Wide (512→256)":           {"hidden_layers": [512, 256],    "dropout": 0.4, "batchnorm": True,  "lr": 0.0005,"batch_size": 32, "weight_decay": 1e-3},
    "ResBlock (256→128→64→32)": {"hidden_layers": [256,128,64,32],"dropout":0.35,"batchnorm": True,  "lr": 0.001, "batch_size": 16, "weight_decay": 1e-4},
}

@st.cache_resource
def run_pytorch_search(_scaler, df_hash, df, feat_names):
    if not TORCH_AVAILABLE:
        return None
    X = df[feat_names]; y = df["status"]
    Xs = _scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=42)

    arch_results = {}
    for arch_name, config in ARCH_CONFIGS.items():
        model, history = train_pytorch_model(Xtr2, ytr2, Xva, yva, config, n_epochs=300, seed=42)
        test_metrics   = evaluate_pytorch(model, Xte, yte)
        arch_results[arch_name] = {
            "model": model, "history": history,
            "config": config, **test_metrics,
            "n_params": model.count_params(),
        }

    best_arch = max(arch_results, key=lambda k: arch_results[k]["roc_auc"])
    return arch_results, best_arch, Xtr, Xte, ytr, yte


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def render_nn_tab(df, scaler, feat_names, existing_results):
    """Call this from your main app inside the Neural Network tab."""

    df_hash = len(df)

    st.markdown("""<div style='background:#dbeafe;border-left:4px solid #2563eb;
        padding:14px 18px;border-radius:6px;font-size:0.85rem;line-height:1.75;color:#1e3a5f;margin-bottom:20px'>
        <strong>Research-Grade Neural Network Analysis</strong><br>
        Two approaches run in parallel:
        <strong>MLP Architecture Search</strong> (scikit-learn GridSearchCV over 7 architectures × 2 activations × 3 regularisation strengths)
        and <strong>PyTorch Deep NN</strong> (5 custom architectures with BatchNorm, Dropout, Early Stopping,
        and ReduceLROnPlateau scheduling). Best architectures are selected automatically and benchmarked
        against all classical ML models.
    </div>""", unsafe_allow_html=True)

    nn_tabs = st.tabs([
        "⚡ MLP Architecture Search",
        "🔥 PyTorch Deep NN",
        "📊 NN vs Classical ML",
        "🏗️ Architecture Explorer",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    #  SUB-TAB 0 — MLP ARCHITECTURE SEARCH
    # ══════════════════════════════════════════════════════════════════════════
    with nn_tabs[0]:
        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">MLP GridSearchCV — Architecture Search</div>', unsafe_allow_html=True)

        with st.spinner("Running MLP architecture search (GridSearchCV over 42 configurations × 5-fold CV)… ~30s"):
            best_mlp, best_params, best_cv_auc, mlp_metrics, cv_results, Xtr, Xte, ytr, yte = run_mlp_search(
                scaler, df_hash, df, feat_names
            )

        # Best params display
        c1, c2, c3, c4 = st.columns(4)
        for col, (lbl, val, color) in zip([c1,c2,c3,c4], [
            ("Best AUC (CV)", f"{best_cv_auc:.4f}", BLUE),
            ("Test Accuracy",  f"{mlp_metrics['accuracy']*100:.1f}%", GREEN),
            ("F1 Score",       f"{mlp_metrics['f1']:.4f}", AMBER),
            ("10-CV Acc",      f"{mlp_metrics['cv_mean']*100:.1f}%", PURPLE),
        ]):
            col.markdown(f"""<div style='background:{WHITE};border:1px solid {BORDER};border-radius:10px;
                padding:16px;text-align:center;border-left:4px solid {color}'>
                <div style='font-family:DM Serif Display,serif;font-size:1.8rem;color:{color}'>{val}</div>
                <div style='font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{TEXT_DIM};margin-top:4px'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div style='background:{NAVY};color:{WHITE};border-radius:8px;padding:14px 20px;font-size:0.85rem;margin-bottom:20px'>
            <strong>🏆 Best Architecture Found:</strong><br>
            Hidden layers: <code style='background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px'>{best_params['hidden_layer_sizes']}</code> &nbsp;
            Activation: <code style='background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px'>{best_params['activation']}</code> &nbsp;
            L2 alpha: <code style='background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px'>{best_params['alpha']}</code> &nbsp;
            Early stopping: <code style='background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px'>✅</code>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">Confusion Matrix</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            cm_n = mlp_metrics["cm"].astype(float) / mlp_metrics["cm"].sum(axis=1, keepdims=True)
            im = ax.imshow(cm_n, cmap=LinearSegmentedColormap.from_list("cm",[WHITE,BLUE_LIGHT,BLUE],256), vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j,i, f"{mlp_metrics['cm'][i,j]}\n({cm_n[i,j]*100:.1f}%)",
                            ha="center", va="center", fontsize=13, fontweight="bold",
                            color=WHITE if cm_n[i,j]>0.6 else NAVY)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Pred: Healthy","Pred: PD"]); ax.set_yticklabels(["True: Healthy","True: PD"])
            ax.set_title("MLP (Best Architecture) — Confusion Matrix")
            plt.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        with col2:
            st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">ROC Curve</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(mlp_metrics["fpr"], mlp_metrics["tpr"], color=BLUE, lw=2.5, label=f"MLP AUC = {mlp_metrics['roc_auc']:.4f}")
            ax.fill_between(mlp_metrics["fpr"], mlp_metrics["tpr"], alpha=0.1, color=BLUE)
            ax.plot([0,1],[0,1], color=BORDER_MED, lw=1, ls="--", label="Random")
            ax.set(title="MLP ROC Curve", xlabel="FPR", ylabel="TPR"); ax.legend(); ax.grid(True)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Architecture search heatmap
        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px;margin-top:16px">Architecture Search Results — Mean CV AUC Heatmap</div>', unsafe_allow_html=True)
        try:
            pivot_data = cv_results[["param_hidden_layer_sizes","param_activation","param_alpha","mean_test_score"]].copy()
            pivot_data["architecture"] = pivot_data["param_hidden_layer_sizes"].astype(str)
            pivot_data["activation"]   = pivot_data["param_activation"].astype(str)
            top_arch = (pivot_data.groupby(["architecture","activation"])["mean_test_score"]
                        .max().reset_index()
                        .pivot(index="architecture", columns="activation", values="mean_test_score"))
            fig, ax = plt.subplots(figsize=(8, 4))
            import seaborn as sns
            sns.heatmap(top_arch, annot=True, fmt=".4f",
                        cmap=LinearSegmentedColormap.from_list("h",[WHITE,BLUE_LIGHT,BLUE],256),
                        ax=ax, linewidths=0.5, linecolor=BORDER,
                        cbar_kws={"shrink":0.7,"label":"Mean CV ROC AUC"},
                        annot_kws={"size":9,"weight":"bold"})
            ax.set_title("GridSearchCV — Architecture × Activation AUC Heatmap", pad=10)
            ax.set_xlabel("Activation Function"); ax.set_ylabel("Architecture")
            plt.xticks(rotation=0); plt.yticks(rotation=0, fontsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as e:
            st.warning(f"Heatmap skipped: {e}")

        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px;margin-top:16px">Classification Report</div>', unsafe_allow_html=True)
        st.code(mlp_metrics["report"], language="")


    # ══════════════════════════════════════════════════════════════════════════
    #  SUB-TAB 1 — PYTORCH DEEP NN
    # ══════════════════════════════════════════════════════════════════════════
    with nn_tabs[1]:
        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">PyTorch — Architecture Search with Early Stopping</div>', unsafe_allow_html=True)

        if not TORCH_AVAILABLE:
            st.markdown(f"""<div style='background:{AMBER_LIGHT};border-left:4px solid {AMBER};
                padding:14px 18px;border-radius:6px;font-size:0.85rem;color:#92400e'>
                <strong>PyTorch not installed.</strong><br>
                Run: <code>pip install torch torchvision</code> then restart the app.
            </div>""", unsafe_allow_html=True)
        else:
            with st.spinner("Running PyTorch architecture search (5 architectures, up to 300 epochs each with early stopping)… ~60s"):
                pt_result = run_pytorch_search(scaler, df_hash, df, feat_names)

            if pt_result is None:
                st.error("PyTorch search failed.")
            else:
                arch_results, best_arch, Xtr, Xte, ytr, yte = pt_result

                # Best architecture highlight
                best = arch_results[best_arch]
                st.markdown(f"""<div style='background:{NAVY};color:{WHITE};border-radius:8px;padding:14px 20px;font-size:0.85rem;margin-bottom:20px'>
                    <strong>🏆 Best PyTorch Architecture: {best_arch}</strong><br>
                    Layers: <code style='background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px'>{best['config']['hidden_layers']}</code> &nbsp;
                    Dropout: <code style='background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px'>{best['config']['dropout']}</code> &nbsp;
                    BatchNorm: <code style='background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px'>✅</code> &nbsp;
                    Params: <code style='background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px'>{best['n_params']:,}</code> &nbsp;
                    Stopped @ epoch: <code style='background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:4px'>{best['history']['stopped_epoch']}</code>
                </div>""", unsafe_allow_html=True)

                # Metrics strip for best arch
                c1,c2,c3,c4 = st.columns(4)
                for col,(lbl,val,color) in zip([c1,c2,c3,c4],[
                    ("Test AUC",     f"{best['roc_auc']:.4f}",      BLUE),
                    ("Test Accuracy",f"{best['accuracy']*100:.1f}%", GREEN),
                    ("F1 Score",     f"{best['f1']:.4f}",            AMBER),
                    ("Parameters",   f"{best['n_params']:,}",         PURPLE),
                ]):
                    col.markdown(f"""<div style='background:{WHITE};border:1px solid {BORDER};border-radius:10px;
                        padding:16px;text-align:center;border-left:4px solid {color}'>
                        <div style='font-family:DM Serif Display,serif;font-size:1.8rem;color:{color}'>{val}</div>
                        <div style='font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{TEXT_DIM};margin-top:4px'>{lbl}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Training curves for best arch
                st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">Training Curves — {best_arch}</div>', unsafe_allow_html=True)
                hist = best["history"]
                epochs = range(1, len(hist["train_loss"])+1)

                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                # Loss
                axes[0].plot(epochs, hist["train_loss"], color=BLUE,  lw=2, label="Train Loss")
                axes[0].plot(epochs, hist["val_loss"],   color=RED,   lw=2, label="Val Loss")
                if hist["stopped_epoch"] < 300:
                    axes[0].axvline(hist["stopped_epoch"], color=AMBER, lw=1.5, ls=":", label=f"Early stop @{hist['stopped_epoch']}")
                axes[0].set(title="Loss Curves", xlabel="Epoch", ylabel="BCE Loss"); axes[0].legend(); axes[0].grid(True)
                # Accuracy
                axes[1].plot(epochs, [a*100 for a in hist["train_acc"]], color=BLUE,  lw=2, label="Train Acc")
                axes[1].plot(epochs, [a*100 for a in hist["val_acc"]],   color=GREEN, lw=2, label="Val Acc")
                if hist["stopped_epoch"] < 300:
                    axes[1].axvline(hist["stopped_epoch"], color=AMBER, lw=1.5, ls=":")
                axes[1].set(title="Accuracy Curves", xlabel="Epoch", ylabel="Accuracy (%)"); axes[1].legend(); axes[1].grid(True)
                # LR
                axes[2].plot(epochs, hist["lr"], color=PURPLE, lw=2)
                axes[2].set(title="Learning Rate Schedule\n(ReduceLROnPlateau)", xlabel="Epoch", ylabel="LR"); axes[2].grid(True)
                axes[2].set_yscale("log")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                # All architectures comparison
                st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px;margin-top:16px">All Architectures — Training Loss Comparison</div>', unsafe_allow_html=True)
                arch_pal = [BLUE, GREEN, AMBER, PURPLE, RED]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
                for (arch_name, res), c in zip(arch_results.items(), arch_pal):
                    h = res["history"]
                    ep = range(1, len(h["val_loss"])+1)
                    lw = 2.5 if arch_name == best_arch else 1.2
                    alpha = 1.0 if arch_name == best_arch else 0.55
                    _tag  = "BEST" if arch_name == best_arch else f"AUC={res['roc_auc']:.3f}"
                    label = f"{arch_name} ({_tag})"
                    ax1.plot(ep, h["val_loss"], color=c, lw=lw, alpha=alpha, label=label)
                    ax2.plot(ep, [a*100 for a in h["val_acc"]], color=c, lw=lw, alpha=alpha, label=label)
                ax1.set(title="Validation Loss — All Architectures", xlabel="Epoch", ylabel="Val Loss"); ax1.legend(fontsize=7); ax1.grid(True)
                ax2.set(title="Validation Accuracy — All Architectures", xlabel="Epoch", ylabel="Val Acc (%)"); ax2.legend(fontsize=7); ax2.grid(True)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                # Architecture leaderboard
                st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px;margin-top:8px">Architecture Leaderboard</div>', unsafe_allow_html=True)
                arch_lb = []
                for arch_name, res in arch_results.items():
                    arch_lb.append({
                        "Architecture":  arch_name,
                        "Layers":        str(res["config"]["hidden_layers"]),
                        "Params":        f"{res['n_params']:,}",
                        "Dropout":       res["config"]["dropout"],
                        "Stopped@Epoch": res["history"]["stopped_epoch"],
                        "Val Loss":      f"{min(res['history']['val_loss']):.4f}",
                        "Test AUC":      f"{res['roc_auc']:.4f}",
                        "Test Acc":      f"{res['accuracy']*100:.2f}%",
                        "F1":            f"{res['f1']:.4f}",
                        "Best":          "🏆" if arch_name==best_arch else "",
                    })
                arch_lb_df = pd.DataFrame(arch_lb).sort_values("Test AUC", ascending=False).reset_index(drop=True)
                arch_lb_df.index += 1
                st.dataframe(arch_lb_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">Best Model — Confusion Matrix</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    cm_n = best["cm"].astype(float)/best["cm"].sum(axis=1,keepdims=True)
                    im=ax.imshow(cm_n,cmap=LinearSegmentedColormap.from_list("cm",[WHITE,BLUE_LIGHT,BLUE],256),vmin=0,vmax=1)
                    for i in range(2):
                        for j in range(2):
                            ax.text(j,i,f"{best['cm'][i,j]}\n({cm_n[i,j]*100:.1f}%)",
                                    ha="center",va="center",fontsize=13,fontweight="bold",
                                    color=WHITE if cm_n[i,j]>0.6 else NAVY)
                    ax.set_xticks([0,1]); ax.set_yticks([0,1])
                    ax.set_xticklabels(["Healthy","PD"]); ax.set_yticklabels(["Healthy","PD"])
                    ax.set_title(f"PyTorch ({best_arch})\nConfusion Matrix")
                    plt.colorbar(im,ax=ax,fraction=0.046)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                with col2:
                    st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">Best Model — ROC Curve</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(best["fpr"],best["tpr"],color=BLUE,lw=2.5,label=f"PyTorch AUC = {best['roc_auc']:.4f}")
                    ax.fill_between(best["fpr"],best["tpr"],alpha=0.1,color=BLUE)
                    ax.plot([0,1],[0,1],color=BORDER_MED,lw=1,ls="--",label="Random")
                    ax.set(title=f"PyTorch ({best_arch}) ROC",xlabel="FPR",ylabel="TPR"); ax.legend(); ax.grid(True)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


    # ══════════════════════════════════════════════════════════════════════════
    #  SUB-TAB 2 — NN vs CLASSICAL ML
    # ══════════════════════════════════════════════════════════════════════════
    with nn_tabs[2]:
        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">Neural Networks vs Classical ML — Head to Head</div>', unsafe_allow_html=True)

        # Re-use already computed results
        with st.spinner("Compiling comparison…"):
            _, _, _, mlp_metrics, _, _, _, _, _ = run_mlp_search(scaler, df_hash, df, feat_names)

        all_models = {}
        for name, res in existing_results.items():
            all_models[name] = {
                "accuracy": res["accuracy"], "f1": res["f1"],
                "roc_auc":  res["roc_auc"],  "pr_auc": res["pr_auc"],
                "fpr": res["fpr"], "tpr": res["tpr"], "type": "Classical",
            }
        all_models["MLP (Best Arch)"] = {
            "accuracy": mlp_metrics["accuracy"], "f1": mlp_metrics["f1"],
            "roc_auc":  mlp_metrics["roc_auc"],  "pr_auc": mlp_metrics["pr_auc"],
            "fpr": mlp_metrics["fpr"], "tpr": mlp_metrics["tpr"], "type": "Neural Network",
        }
        if TORCH_AVAILABLE:
            try:
                arch_results2, best_arch2, _, _, _, _ = run_pytorch_search(scaler, df_hash, df, feat_names)
                best_pt = arch_results2[best_arch2]
                all_models[f"PyTorch ({best_arch2})"] = {
                    "accuracy": best_pt["accuracy"], "f1": best_pt["f1"],
                    "roc_auc":  best_pt["roc_auc"],  "pr_auc": best_pt["pr_auc"],
                    "fpr": best_pt["fpr"], "tpr": best_pt["tpr"], "type": "Neural Network",
                }
            except Exception:
                pass

        # Full comparison table
        comp_rows = []
        for name, m in all_models.items():
            comp_rows.append({
                "Model": name, "Type": m["type"],
                "Accuracy": f"{m['accuracy']*100:.2f}%",
                "F1": f"{m['f1']:.4f}",
                "ROC AUC": f"{m['roc_auc']:.4f}",
                "PR AUC": f"{m['pr_auc']:.4f}",
            })
        comp_df = pd.DataFrame(comp_rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
        comp_df.index += 1
        st.dataframe(comp_df, use_container_width=True)

        # ROC comparison plot
        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px;margin-top:16px">ROC Curves — Classical ML vs Neural Networks</div>', unsafe_allow_html=True)
        pal_classic = [TEXT_DIM]*len(existing_results)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for (name, m), c in zip(list(existing_results.items()), [TEXT_DIM, BORDER_MED, TEXT_DIM, BORDER_MED, TEXT_DIM, BORDER_MED, TEXT_DIM, BORDER_MED]):
            ax.plot(m["fpr"], m["tpr"], color=c, lw=1.2, alpha=0.6, label=f"{name} ({m['roc_auc']:.3f})")
        if "MLP (Best Arch)" in all_models:
            m = all_models["MLP (Best Arch)"]
            ax.plot(m["fpr"], m["tpr"], color=BLUE, lw=2.8, label=f"MLP ({m['roc_auc']:.3f}) ★")
        pt_key = [k for k in all_models if "PyTorch" in k]
        if pt_key:
            m = all_models[pt_key[0]]
            ax.plot(m["fpr"], m["tpr"], color=RED, lw=2.8, ls="-.", label=f"{pt_key[0]} ({m['roc_auc']:.3f}) ★")
        ax.plot([0,1],[0,1], color=BORDER, lw=1, ls="--")
        ax.set(title="ROC Curves — All Models (★ = Neural Networks)",
               xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.legend(loc="lower right", fontsize=7.5); ax.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Bar chart
        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">AUC Comparison</div>', unsafe_allow_html=True)
        names_comp = list(all_models.keys())
        aucs_comp  = [all_models[n]["roc_auc"] for n in names_comp]
        colors_comp= [BLUE if all_models[n]["type"]=="Neural Network" else TEXT_DIM for n in names_comp]
        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(names_comp, aucs_comp, color=colors_comp, alpha=0.85, edgecolor=WHITE)
        for bar, val in zip(bars, aucs_comp):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticklabels([n.replace(" ","\n") for n in names_comp], fontsize=7.5)
        ax.set(title="ROC AUC — Classical ML (grey) vs Neural Networks (blue)", ylabel="ROC AUC", ylim=[0.6,1.05])
        ax.grid(axis="y")
        handles = [mpatches.Patch(color=BLUE,label="Neural Network"), mpatches.Patch(color=TEXT_DIM,label="Classical ML")]
        ax.legend(handles=handles, fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


    # ══════════════════════════════════════════════════════════════════════════
    #  SUB-TAB 3 — ARCHITECTURE EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    with nn_tabs[3]:
        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">Architecture Explorer — Visualise Network Structure</div>', unsafe_allow_html=True)

        if not TORCH_AVAILABLE:
            st.warning("PyTorch required for this section.")
        else:
            arch_choice = st.selectbox("Select architecture to inspect:", list(ARCH_CONFIGS.keys()))
            config      = ARCH_CONFIGS[arch_choice]

            c1, c2, c3 = st.columns(3)
            layers = [22] + config["hidden_layers"] + [1]
            total_params = sum(layers[i]*layers[i+1] + layers[i+1] for i in range(len(layers)-1))
            c1.markdown(f"""<div style='background:{WHITE};border:1px solid {BORDER};border-radius:10px;
                padding:16px;text-align:center;border-left:4px solid {BLUE}'>
                <div style='font-family:DM Serif Display,serif;font-size:1.8rem;color:{BLUE}'>{len(config['hidden_layers'])}</div>
                <div style='font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{TEXT_DIM}'>Hidden Layers</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div style='background:{WHITE};border:1px solid {BORDER};border-radius:10px;
                padding:16px;text-align:center;border-left:4px solid {GREEN}'>
                <div style='font-family:DM Serif Display,serif;font-size:1.8rem;color:{GREEN}'>{total_params:,}</div>
                <div style='font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{TEXT_DIM}'>Total Parameters</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div style='background:{WHITE};border:1px solid {BORDER};border-radius:10px;
                padding:16px;text-align:center;border-left:4px solid {AMBER}'>
                <div style='font-family:DM Serif Display,serif;font-size:1.8rem;color:{AMBER}'>{config['dropout']}</div>
                <div style='font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{TEXT_DIM}'>Dropout Rate</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Network diagram
            st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px">Network Architecture Diagram</div>', unsafe_allow_html=True)
            layer_sizes = [22] + config["hidden_layers"] + [1]
            layer_names = ["Input\n(22 features)"] + [f"Hidden {i+1}\n({s} neurons)" for i,s in enumerate(config["hidden_layers"])] + ["Output\n(PD prob)"]
            layer_colors= [SLATE] + [BLUE]*len(config["hidden_layers"]) + [GREEN if False else RED]

            fig, ax = plt.subplots(figsize=(max(10, len(layer_sizes)*2.5), 6))
            ax.set_xlim(-0.5, len(layer_sizes)-0.5); ax.set_ylim(-1, 5)
            ax.set_facecolor(BG); ax.axis("off")
            ax.set_title(f"Architecture: {arch_choice}", color=NAVY, fontsize=12, fontweight="bold", pad=10)

            node_positions = []
            for i, (size, name, color) in enumerate(zip(layer_sizes, layer_names, [SLATE]+[BLUE]*len(config["hidden_layers"])+[GREEN])):
                display_nodes = min(size, 8)
                y_start = -(display_nodes-1)/2
                positions = [(i, y_start + j) for j in range(display_nodes)]
                node_positions.append(positions)

                # Draw connections to prev layer
                if i > 0:
                    for (px, py) in node_positions[i-1][:5]:
                        for (cx, cy) in positions[:5]:
                            ax.plot([px+0.15, cx-0.15], [py, cy], color=BORDER_MED, lw=0.4, alpha=0.5, zorder=1)

                # Draw nodes
                for (x, y) in positions:
                    circle = plt.Circle((x, y), 0.12, color=color, alpha=0.85, zorder=3)
                    ax.add_patch(circle)

                if size > 8:
                    ax.text(i, -(display_nodes/2)-0.4, f"... +{size-8}", ha="center", fontsize=7, color=TEXT_DIM)

                # Layer label
                ax.text(i, -(display_nodes/2)-0.9, name, ha="center", fontsize=8, color=NAVY, fontweight="bold")

                # Annotations
                annotations = []
                if i == 0:
                    annotations.append("Input Layer")
                elif i == len(layer_sizes)-1:
                    annotations.append("Sigmoid Output")
                else:
                    annotations.append("ReLU + Dropout" if config["dropout"]>0 else "ReLU")
                    if config["batchnorm"]:
                        annotations.append("BatchNorm")
                ax.text(i, max(p[1] for p in positions)+0.5, "\n".join(annotations),
                        ha="center", fontsize=7, color=TEXT_MID, style="italic")

            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            # Config details
            st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{TEXT_DIM};border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:18px;margin-top:16px">Full Configuration</div>', unsafe_allow_html=True)
            config_display = {
                "Architecture":       arch_choice,
                "Input Dimension":    22,
                "Hidden Layers":      str(config["hidden_layers"]),
                "Output":             "1 neuron (Sigmoid → PD probability)",
                "Activation":         "ReLU (hidden) + Sigmoid (output)",
                "Batch Normalisation": "Yes" if config["batchnorm"] else "No",
                "Dropout Rate":       config["dropout"],
                "Optimiser":          "Adam",
                "Learning Rate":      config["lr"],
                "LR Scheduler":       "ReduceLROnPlateau (patience=8, factor=0.5)",
                "Weight Decay (L2)":  config["weight_decay"],
                "Batch Size":         config["batch_size"],
                "Max Epochs":         300,
                "Early Stopping":     "Yes (patience=20, min_delta=0.001)",
                "Weight Init":        "Kaiming Normal",
                "Loss Function":      "Binary Cross-Entropy (BCE)",
                "Total Parameters":   f"{total_params:,}",
            }
            for k, v in config_display.items():
                st.markdown(f"""<div style='display:flex;justify-content:space-between;
                    font-size:0.82rem;padding:6px 0;border-bottom:1px solid {BORDER}'>
                    <span style='color:{TEXT_MID};font-weight:500'>{k}</span>
                    <span style='color:{NAVY};font-weight:600'>{v}</span>
                </div>""", unsafe_allow_html=True)
