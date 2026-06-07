# interpret_features.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    accuracy_score,
    f1_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    
from sklearn.model_selection import GroupKFold, KFold

ML_DATA_FOLDER = "./features/optimal/final"
RESULTS_FOLDER = "./results/optimal"

# ==========================================
# CONFIG: Toggle session-aware splitting here
# ==========================================
SESSION_AWARE = False   # set to True to enable session-aware CV


LABEL_COLS = [
    "heel_lift", "toe_out", "toe_in", "weight_shift_to_toes",
    "knee_valgus", "knee_varus", "excessive_knee_travel",
    "butt_wink", "lumbar_extension", "hip_shift",
    "forward_lean", "chest_collapse",
    "shallow_depth", "excessive_depth",
    "weak_bracing", "poor_breathing",
    "dropping_too_fast", "unintentional_pause",
    "asymmetrical_pattern", "poor_stance",
    "good_technique",
]

def load_saved_models(variant, model_type="single"):
    """
    model_type: "single" or "multi"
    Returns a dict: { model_name: model_object }
    """

    models = {}

    if model_type == "single":
        model_dir = "saved_models"
        for fault in LABEL_COLS:
            for model_name in ["XGBoost", "LightGBM", "RandomForest", "MLP"]:
                path = os.path.join(model_dir, f"{variant}_{fault}_{model_name}.pkl")
                if os.path.exists(path):
                    models[(fault, model_name)] = joblib.load(path)

    elif model_type == "multi":
        model_dir = "saved_models_multilabel"
        for model_name in ["XGBoost Multi", "LightGBM Multi", "RandomForest Multi", "MLP Multi"]:
            path = os.path.join(model_dir, f"{variant}_{model_name}.pkl")
            if os.path.exists(path):
                models[model_name] = joblib.load(path)

    return models


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop(columns=LABEL_COLS + ["session_id"])
    y = df[LABEL_COLS]
    sessions = df["session_id"].values
    return X, y.values, sessions


def session_aware_split(X, y, sessions, seed=42):
    unique_sessions = np.unique(sessions)

    train_sess, test_sess = train_test_split(
        unique_sessions, test_size=0.2, random_state=seed
    )
    train_sess, val_sess = train_test_split(
        train_sess, test_size=0.2, random_state=seed
    )

    def mask(sess_ids):
        return np.isin(sessions, sess_ids)

    train_mask = mask(train_sess)
    val_mask = mask(val_sess)
    test_mask = mask(test_sess)

    return (
        X.iloc[train_mask], y[train_mask],
        X.iloc[val_mask], y[val_mask],
        X.iloc[test_mask], y[test_mask],
        train_sess, val_sess, test_sess
    )


def build_model_xgb():
    return OneVsRestClassifier(XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", tree_method="hist", n_jobs=-1
    ))

def build_model_lgbm():
    return OneVsRestClassifier(LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=50, subsample=0.9,
        colsample_bytree=0.9, n_jobs=-1
    ))

def build_model_rf():
    return OneVsRestClassifier(RandomForestClassifier(
        n_estimators=500, max_depth=None, n_jobs=-1
    ))

def build_model_mlp():
    return OneVsRestClassifier(MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=300
    ))


def build_models():
    return [
        ("XGBoost", OneVsRestClassifier(XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", tree_method="hist", n_jobs=-1
        ))),
        ("LightGBM", OneVsRestClassifier(LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=50, subsample=0.9,
            colsample_bytree=0.9, n_jobs=-1
        ))),
        ("RandomForest", OneVsRestClassifier(RandomForestClassifier(
            n_estimators=500, max_depth=None, n_jobs=-1
        ))),
        ("MLP", OneVsRestClassifier(MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=300
        ))),
    ]


def plot_confusion_matrices(y_true, y_pred, labels, out_dir, model_name):
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    for i, label in enumerate(labels):
        tn, fp, fn, tp = mcm[i].ravel()
        mat = np.array([[tn, fp], [fn, tp]], dtype=float)
        # row-normalise
        mat = mat / mat.sum(axis=1, keepdims=True).clip(min=1e-9)

        plt.figure(figsize=(4, 3))
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        plt.title(f"{model_name} — {label}")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{model_name}_confmat_{label}.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_feature_importance_tree(model, feature_names, out_dir, model_name, top_n=20):
    """
    Handles both:
    - OneVsRestClassifier with estimators_
    - Single tree-based models (XGBClassifier, LGBMClassifier, RandomForestClassifier)
    """

    # Case 1: OneVsRestClassifier
    if hasattr(model, "estimators_"):
        for i, label in enumerate(LABEL_COLS):
            est = model.estimators_[i]
            if not hasattr(est, "feature_importances_"):
                continue

            importances = est.feature_importances_
            idx = np.argsort(importances)[::-1][:top_n]

            plt.figure(figsize=(6, 5))
            sns.barplot(
                x=importances[idx],
                y=[feature_names[j] for j in idx],
                orient="h"
            )
            plt.title(f"{model_name} — Top {top_n} features for {label}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model_name}_featimp_{label}.png"), dpi=200)
            plt.close()

        return

    # Case 2: Single classifier (your saved single-label models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(6, 5))
        sns.barplot(
            x=importances[idx],
            y=[feature_names[j] for j in idx],
            orient="h"
        )
        plt.title(f"{model_name} — Top {top_n} features")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_featimp.png"), dpi=200)
        plt.close()

        return

    print(f"[WARN] No feature_importances_ available for model: {model_name}")



def shap_summary_tree(model, X_sample, feature_names, out_dir, model_name, fault="knee_valgus"):
    # Skip MLP entirely
    if model_name == "MLP":
        return

    # Case 1 — OneVsRestClassifier (multi-label models)
    if hasattr(model, "estimators_"):
        idx_fault = LABEL_COLS.index(fault)
        est = model.estimators_[idx_fault]

        from sklearn.dummy import DummyClassifier
        if isinstance(est, DummyClassifier):
            print(f"[SHAP] Skipping {model_name} / {fault} — constant predictor.")
            return

        try:
            explainer = shap.TreeExplainer(est)
        except Exception as e:
            print(f"[SHAP] Skipping {model_name} / {fault} — unsupported model: {e}")
            return

        shap_values = explainer.shap_values(X_sample)

        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        fname = os.path.join(out_dir, f"{model_name}_shap_{fault}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        return

    # Case 2 — Single classifier (your saved single-label models)
    if hasattr(model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            print(f"[SHAP] Skipping {model_name} — unsupported model: {e}")
            return

        shap_values = explainer.shap_values(X_sample)

        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        fname = os.path.join(out_dir, f"{model_name}_shap.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        return

    print(f"[SHAP] No SHAP support for model: {model_name}")



def permutation_importance_mlp(model, X_test_s, y_test, feature_names, out_dir, model_name, top_n=20):
    print(f"[MLP] Computing permutation importance...")

    # Case 1 — OneVsRestClassifier (multi-label)
    if hasattr(model, "estimators_"):
        # Compute importance for each fault separately
        for i, fault in enumerate(LABEL_COLS):
            est = model.estimators_[i]
            result = permutation_importance(
                est,
                X_test_s,
                y_test[:, i],
                scoring="accuracy",
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            importances = result.importances_mean
            idx = np.argsort(importances)[::-1][:top_n]

            plt.figure(figsize=(6, 5))
            sns.barplot(
                x=importances[idx],
                y=[feature_names[j] for j in idx],
                orient="h"
            )
            plt.title(f"{model_name} — Permutation Importance ({fault})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model_name}_permimp_{fault}.png"), dpi=200)
            plt.close()

        return

    # Case 2 — Single-label MLP
    result = permutation_importance(
        model,
        X_test_s,
        y_test,
        scoring="accuracy",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    model.perm_importances_ = result.importances_mean
    importances = result.importances_mean
    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(6, 5))
    sns.barplot(
        x=importances[idx],
        y=[feature_names[j] for j in idx],
        orient="h"
    )
    plt.title(f"{model_name} — Permutation Importance (Top {top_n})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_perm_importance.png"), dpi=200)
    plt.close()



def plot_feature_distributions(X, y, feature_names, out_dir, variant, fault, top_features):
    fault_idx = LABEL_COLS.index(fault)
    labels = y[:, fault_idx]  # 0/1

    dist_dir = os.path.join(out_dir, "feature_distributions")
    os.makedirs(dist_dir, exist_ok=True)

    df = pd.DataFrame(X, columns=feature_names)
    df["fault"] = labels

    for feat in top_features:
        plt.figure(figsize=(6, 4))
        sns.violinplot(
            data=df,
            x="fault",
            y=feat,
            palette="coolwarm",
            cut=0
        )
        plt.title(f"{variant} — {feat} distribution by {fault}")
        plt.tight_layout()
        fname = os.path.join(dist_dir, f"violin_{feat}_{fault}.png")
        plt.savefig(fname, dpi=200)
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(
            data=df,
            x="fault",
            y=feat,
            palette="coolwarm"
        )
        plt.title(f"{variant} — {feat} boxplot by {fault}")
        plt.tight_layout()
        fname = os.path.join(dist_dir, f"box_{feat}_{fault}.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_embeddings(X_s, y, feature_names, out_dir, variant, fault="knee_valgus"):
    # X_s: scaled features (numpy array)
    # y:   labels (numpy array, shape [n_samples, n_labels])

    if fault not in LABEL_COLS:
        print(f"[EMBED] Fault {fault} not in LABEL_COLS, skipping embeddings.")
        return

    fault_idx = LABEL_COLS.index(fault)
    labels = y[:, fault_idx]  # 0/1 for that fault

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_s)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=labels,
        palette="coolwarm",
        alpha=0.7,
        s=30,
    )
    plt.title(f"{variant} — PCA (colored by {fault})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title=fault)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"embed_PCA_{fault}.png")
    plt.savefig(fname, dpi=200)
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(X_s)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=labels,
        palette="coolwarm",
        alpha=0.7,
        s=30,
    )
    plt.title(f"{variant} — t-SNE (colored by {fault})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title=fault)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"embed_TSNE_{fault}.png")
    plt.savefig(fname, dpi=200)
    plt.close()

    # UMAP (if available)
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_s)

        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            hue=labels,
            palette="coolwarm",
            alpha=0.7,
            s=30,
        )
        plt.title(f"{variant} — UMAP (colored by {fault})")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(title=fault)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"embed_UMAP_{fault}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
    else:
        print("[EMBED] UMAP not installed, skipping UMAP embeddings.")


def plot_all_embeddings(X_s, y, feature_names, out_dir, variant):
    """
    Compute PCA, t-SNE, UMAP ONCE, then generate
    embeddings for ALL faults by recolouring.
    """

    # -------------------------
    # Compute embeddings ONCE
    # -------------------------
    print("[EMBED] Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_s)

    print("[EMBED] Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(X_s)

    X_umap = None
    if HAS_UMAP:
        print("[EMBED] Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_s)
        
    # -------------------------
    # Multi-fault embedding grids
    # -------------------------
    print("[EMBED] Creating multi-fault grids...")
    
    plot_embedding_grid(
        X_pca,
        y,
        LABEL_COLS,
        title=f"{variant} — PCA Multi-Fault Grid",
        out_path=os.path.join(out_dir, "grid_PCA_all_faults.png")
    )
    
    plot_embedding_grid(
        X_tsne,
        y,
        LABEL_COLS,
        title=f"{variant} — t-SNE Multi-Fault Grid",
        out_path=os.path.join(out_dir, "grid_TSNE_all_faults.png")
    )
    
    if X_umap is not None:
        plot_embedding_grid(
            X_umap,
            y,
            LABEL_COLS,
            title=f"{variant} — UMAP Multi-Fault Grid",
            out_path=os.path.join(out_dir, "grid_UMAP_all_faults.png")
        )

    # -------------------------
    # Generate plots for ALL faults
    # -------------------------
    for fault in LABEL_COLS:
        fault_idx = LABEL_COLS.index(fault)
        labels = y[:, fault_idx]

        # PCA
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1],
            hue=labels, palette="coolwarm", alpha=0.7, s=30
        )
        plt.title(f"{variant} — PCA coloured by {fault}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"embed_PCA_{fault}.png"), dpi=200)
        plt.close()

        # t-SNE
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            x=X_tsne[:, 0], y=X_tsne[:, 1],
            hue=labels, palette="coolwarm", alpha=0.7, s=30
        )
        plt.title(f"{variant} — t-SNE coloured by {fault}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"embed_TSNE_{fault}.png"), dpi=200)
        plt.close()

        # UMAP
        if X_umap is not None:
            plt.figure(figsize=(6, 5))
            sns.scatterplot(
                x=X_umap[:, 0], y=X_umap[:, 1],
                hue=labels, palette="coolwarm", alpha=0.7, s=30
            )
            plt.title(f"{variant} — UMAP coloured by {fault}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"embed_UMAP_{fault}.png"), dpi=200)
            plt.close()


def plot_embedding_grid(X_emb, y, faults, title, out_path, n_cols=7):
    """
    Create a grid of subplots for all faults using a precomputed embedding.
    X_emb: (n_samples, 2)
    y: (n_samples, n_faults)
    faults: list of fault names
    """

    n_faults = len(faults)
    n_rows = int(np.ceil(n_faults / n_cols))

    plt.figure(figsize=(3*n_cols, 3*n_rows))

    for i, fault in enumerate(faults):
        fault_idx = LABEL_COLS.index(fault)
        labels = y[:, fault_idx]

        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.scatter(
            X_emb[:, 0], X_emb[:, 1],
            c=labels, cmap="coolwarm", s=8, alpha=0.7
        )
        ax.set_title(fault, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=300)
    plt.close()
    

def compute_3d_embeddings(X_s):
    """Compute PCA3 and UMAP3 (if available) once."""
    print("[EMBED-3D] Computing PCA 3D...")
    pca3 = PCA(n_components=3, random_state=42)
    X_pca3 = pca3.fit_transform(X_s)

    X_umap3 = None
    if HAS_UMAP:
        print("[EMBED-3D] Computing UMAP 3D...")
        reducer3 = umap.UMAP(n_components=3, random_state=42)
        X_umap3 = reducer3.fit_transform(X_s)

    return X_pca3, X_umap3, pca3


def plot_3d_static_per_fault(X_emb3, y, variant, out_dir, method_name):
    """
    X_emb3: (n_samples, 3)
    y: (n_samples, n_labels)
    One static 3D PNG per fault.
    """
    for fault in LABEL_COLS:
        fault_idx = LABEL_COLS.index(fault)
        labels = y[:, fault_idx]

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            X_emb3[:, 0], X_emb3[:, 1], X_emb3[:, 2],
            c=labels, cmap="coolwarm", s=10, alpha=0.7
        )
        ax.set_title(f"{variant} — {method_name} 3D — {fault}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.tight_layout()
        fname = os.path.join(out_dir, f"embed3D_{method_name}_{fault}.png")
        plt.savefig(fname, dpi=250)
        plt.close()


def plot_3d_interactive_per_fault(X_emb3, y, variant, out_dir, method_name):
    """
    One interactive HTML 3D scatter per fault.
    """
    inter_dir = os.path.join(out_dir, "interactive_3d")
    os.makedirs(inter_dir, exist_ok=True)

    for fault in LABEL_COLS:
        fault_idx = LABEL_COLS.index(fault)
        labels = y[:, fault_idx]

        fig = px.scatter_3d(
            x=X_emb3[:, 0],
            y=X_emb3[:, 1],
            z=X_emb3[:, 2],
            color=labels.astype(str),
            opacity=0.7,
            title=f"{variant} — {method_name} 3D — {fault}",
        )
        fig.update_traces(marker=dict(size=3))
        out_html = os.path.join(inter_dir, f"embed3D_{method_name}_{fault}.html")
        fig.write_html(out_html)


def get_top_k_features(model_name, model, feature_names, k=10):
    # Tree models
    if model_name in ["XGBoost", "LightGBM", "RandomForest"]:
        importances = np.zeros(len(feature_names))
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                importances += est.feature_importances_
        idx = np.argsort(importances)[::-1][:k]
        return [feature_names[i] for i in idx]

    # MLP — use permutation importance stored earlier
    if model_name == "MLP" and hasattr(model, "perm_importances_"):
        importances = model.perm_importances_
        idx = np.argsort(importances)[::-1][:k]
        return [feature_names[i] for i in idx]

    return []


from sklearn.model_selection import GroupKFold, KFold

def cross_validate_model(model_name, model_builder, X, y, groups, feature_names, out_dir, variant, n_splits=5):
    if groups is not None:
        print(f"[CV] Running {n_splits}-fold SESSION-AWARE CV for {model_name}...")
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        print(f"[CV] Running {n_splits}-fold RANDOM CV for {model_name}...")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_builder()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)

        fold_results.append({
            "fold": fold_idx + 1,
            "macro_f1": macro_f1,
            "accuracy": acc
        })

        print(f"  Fold {fold_idx+1}: macro-F1={macro_f1:.4f}, acc={acc:.4f}")

    df = pd.DataFrame(fold_results)
    df["macro_f1_mean"] = df["macro_f1"].mean()
    df["macro_f1_std"] = df["macro_f1"].std()
    df["accuracy_mean"] = df["accuracy"].mean()
    df["accuracy_std"] = df["accuracy"].std()

    out_csv = os.path.join(out_dir, f"cv_results_{model_name}.csv")
    df.to_csv(out_csv, index=False)

    print(f"[CV] Saved CV results to: {out_csv}")

    return df

def compute_fault_cooccurrence(y, labels):
    """
    y: numpy array (n_samples, n_faults)
    labels: list of fault names
    Returns:
        co_raw: raw co-occurrence counts
        co_norm: normalised co-occurrence (P(B|A))
    """
    n_faults = len(labels)
    co_raw = np.zeros((n_faults, n_faults), dtype=int)

    # Raw co-occurrence counts
    for i in range(n_faults):
        for j in range(n_faults):
            co_raw[i, j] = np.sum((y[:, i] == 1) & (y[:, j] == 1))

    # Normalised co-occurrence: P(fault_j | fault_i)
    support = co_raw.diagonal().astype(float)
    support[support == 0] = 1e-9  # avoid division by zero
    co_norm = co_raw / support[:, None]

    return co_raw, co_norm


def plot_fault_cooccurrence(co_matrix, labels, out_path, title="Fault Co-occurrence"):
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        co_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Reds",
        annot=False
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def compute_fault_support(y):
    return y.sum(axis=0)  # shape: (n_faults,)


def compute_confusion_difficulty(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    difficulties = []
    for i in range(y_true.shape[1]):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i]).ravel()
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)
        ber = 0.5 * (fpr + fnr)  # balanced error rate
        difficulties.append(ber)
    return np.array(difficulties)


from sklearn.metrics import silhouette_score

def compute_embedding_difficulty(X_emb, y):
    difficulties = []
    for i in range(y.shape[1]):
        labels = y[:, i]
        if labels.sum() < 2 or (labels == 0).sum() < 2:
            difficulties.append(1.0)  # maximally hard
            continue
        try:
            score = silhouette_score(X_emb, labels)
            difficulties.append(1 - score)  # lower silhouette = harder
        except:
            difficulties.append(1.0)
    return np.array(difficulties)


from scipy.stats import pearsonr

def compute_pca_loadings(pca_model, feature_names, out_dir, variant, top_n=15):
    """
    pca_model: fitted PCA with n_components=3
    """
    loadings = pca_model.components_.T  # shape: (n_features, 3)
    df = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=["PC1", "PC2", "PC3"]
    )
    df.to_csv(os.path.join(out_dir, f"{variant}_pca3_loadings.csv"))

    # Plot top features per PC
    for comp_idx, comp_name in enumerate(["PC1", "PC2", "PC3"]):
        comp = loadings[:, comp_idx]
        idx = np.argsort(np.abs(comp))[::-1][:top_n]
        plt.figure(figsize=(7, 5))
        sns.barplot(
            x=comp[idx],
            y=[feature_names[i] for i in idx],
            orient="h",
            palette="coolwarm"
        )
        plt.title(f"{variant} — Top {top_n} features for {comp_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{variant}_pca3_{comp_name}_top_features.png"), dpi=250)
        plt.close()


def compute_embedding_feature_correlations(X_emb, X_orig, feature_names, out_dir, variant, prefix):
    """
    X_emb: (n_samples, d) embedding (PCA3, UMAP3, etc.)
    X_orig: (n_samples, n_features) scaled original features
    """
    n_features = X_orig.shape[1]
    n_dims = X_emb.shape[1]

    rows = []
    for fi in range(n_features):
        feat_vals = X_orig[:, fi]
        for di in range(n_dims):
            emb_vals = X_emb[:, di]
            r, p = pearsonr(feat_vals, emb_vals)
            rows.append({
                "feature": feature_names[fi],
                "dim": di,
                "r": r,
                "p": p
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{variant}_{prefix}_feature_correlations.csv"), index=False)

    # Optional: heatmap of |r|
    pivot = df.pivot(index="feature", columns="dim", values="r").abs()
    plt.figure(figsize=(8, max(6, 0.25 * len(feature_names))))
    sns.heatmap(pivot, cmap="viridis", center=0)
    plt.title(f"{variant} — |Correlation| between features and {prefix} dims")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{variant}_{prefix}_feature_correlations_heatmap.png"), dpi=250)
    plt.close()


def compute_fault_difficulty(
    macro_f1,
    confusion_diff,
    support,
    cooccur_norm,
    embed_diff,
    labels
):
    # Normalise components
    macro_f1_norm = (macro_f1 - macro_f1.min()) / (macro_f1.max() - macro_f1.min() + 1e-9)
    support_norm = 1 - (support / support.max())
    cooccur_norm = (cooccur_norm - cooccur_norm.min()) / (cooccur_norm.max() - cooccur_norm.min() + 1e-9)
    embed_norm = (embed_diff - embed_diff.min()) / (embed_diff.max() - embed_diff.min() + 1e-9)

    # Weighted difficulty
    difficulty = (
        0.40 * (1 - macro_f1_norm) +
        0.20 * confusion_diff +
        0.10 * support_norm +
        0.15 * cooccur_norm +
        0.15 * embed_norm
    )

    return pd.DataFrame({
        "fault": labels,
        "difficulty": difficulty,
        "macro_f1": macro_f1,
        "confusion": confusion_diff,
        "support": support,
        "cooccur": cooccur_norm,
        "embedding": embed_norm
    }).sort_values("difficulty", ascending=False)


from sklearn.neighbors import NearestNeighbors

def compute_cluster_purity(X_emb, y, k=10):
    """
    X_emb: (n_samples, d)
    y: (n_samples,) binary labels for a single fault
    Returns: purity score in [0,1]
    """
    positives = np.where(y == 1)[0]
    if len(positives) < 2:
        return 0.0

    nbrs = NearestNeighbors(n_neighbors=min(k, len(y))).fit(X_emb)
    distances, indices = nbrs.kneighbors(X_emb[positives])

    purity_scores = []
    for neigh_idx in indices:
        neigh_labels = y[neigh_idx]
        purity = np.mean(neigh_labels == 1)
        purity_scores.append(purity)

    return np.mean(purity_scores)


def compute_nn_separability(X_emb, y):
    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]

    if len(positives) < 2 or len(negatives) < 2:
        return 0.0

    nbr_pos = NearestNeighbors(n_neighbors=1).fit(X_emb[positives])
    nbr_neg = NearestNeighbors(n_neighbors=1).fit(X_emb[negatives])

    sep_scores = []
    for idx in positives:
        d_pos, _ = nbr_pos.kneighbors([X_emb[idx]], n_neighbors=1)
        d_neg, _ = nbr_neg.kneighbors([X_emb[idx]], n_neighbors=1)
        sep_scores.append(float(d_neg - d_pos))

    return np.mean(sep_scores)


def compute_density_ratio(X_emb, y, bandwidth=0.5):
    from sklearn.neighbors import KernelDensity

    positives = X_emb[y == 1]
    negatives = X_emb[y == 0]

    if len(positives) < 2 or len(negatives) < 2:
        return 0.0

    kde_pos = KernelDensity(bandwidth=bandwidth).fit(positives)
    kde_neg = KernelDensity(bandwidth=bandwidth).fit(negatives)

    # Evaluate density at positive samples
    log_d_pos = kde_pos.score_samples(positives)
    log_d_neg = kde_neg.score_samples(positives)

    ratio = np.exp(log_d_pos - log_d_neg)
    return np.mean(ratio)


def compute_cluster_purity_metrics(X_emb, y, labels):
    purity_list = []
    nn_sep_list = []
    density_list = []
    silhouette_list = []

    for i, fault in enumerate(labels):
        y_fault = y[:, i]

        # Silhouette
        if y_fault.sum() > 1 and (y_fault == 0).sum() > 1:
            try:
                sil = silhouette_score(X_emb, y_fault)
            except:
                sil = -1
        else:
            sil = -1

        silhouette_list.append(sil)

        # Purity
        purity = compute_cluster_purity(X_emb, y_fault)
        purity_list.append(purity)

        # NN separability
        nn_sep = compute_nn_separability(X_emb, y_fault)
        nn_sep_list.append(nn_sep)

        # Density ratio
        dens = compute_density_ratio(X_emb, y_fault)
        density_list.append(dens)

    return pd.DataFrame({
        "fault": labels,
        "silhouette": silhouette_list,
        "purity": purity_list,
        "nn_separability": nn_sep_list,
        "density_ratio": density_list
    }).sort_values("purity", ascending=False)




def run_for_variant(dataset_path):
    base = os.path.basename(dataset_path)
    variant = base[len("final_features_"):-len(".csv")]
    print(f"\n=== INTERPRETABILITY: {variant} ===")

    out_dir = os.path.join(RESULTS_FOLDER, variant)
    os.makedirs(out_dir, exist_ok=True)

    X, y, sessions = load_dataset(dataset_path)
    feature_names = X.columns

    # -----------------------------------------
    # TRAIN/VAL/TEST SPLIT
    # -----------------------------------------
    if SESSION_AWARE:
        print(">>> Using SESSION-AWARE split")
        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            train_sess, val_sess, test_sess
        ) = session_aware_split(X, y, sessions)

    else:
        print(">>> Using RANDOM split")
        # 60% train, 20% val, 20% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, shuffle=True
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
        )

    # -----------------------------------------
    # SCALING
    # -----------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Embeddings use full dataset
    X_embed = np.vstack([X_train_s, X_val_s, X_test_s])
    y_embed = np.vstack([y_train, y_val, y_test])
    
    
    # -------------------------
    # FAULT CO-OCCURRENCE MAPS
    # -------------------------
    co_raw, co_norm = compute_fault_cooccurrence(y_embed, LABEL_COLS)
    
    plot_fault_cooccurrence(
        co_raw,
        LABEL_COLS,
        os.path.join(out_dir, "fault_cooccurrence_raw.png"),
        title=f"{variant} — Raw Fault Co-occurrence"
    )
    
    plot_fault_cooccurrence(
        co_norm,
        LABEL_COLS,
        os.path.join(out_dir, "fault_cooccurrence_normalised.png"),
        title=f"{variant} — Normalised Fault Co-occurrence (P(B|A))"
    )


    # -----------------------------------------
    # 3D EMBEDDINGS
    # -----------------------------------------
    X_pca3, X_umap3, pca3_model = compute_3d_embeddings(X_embed)

    compute_pca_loadings(pca3_model, feature_names, out_dir, variant)
    compute_embedding_feature_correlations(X_pca3, X_embed, feature_names, out_dir, variant, prefix="PCA3")
    if X_umap3 is not None:
        compute_embedding_feature_correlations(X_umap3, X_embed, feature_names, out_dir, variant, prefix="UMAP3")


    plot_3d_static_per_fault(X_pca3, y_embed, variant, out_dir, method_name="PCA3")
    if X_umap3 is not None:
        plot_3d_static_per_fault(X_umap3, y_embed, variant, out_dir, method_name="UMAP3")

    plot_3d_interactive_per_fault(X_pca3, y_embed, variant, out_dir, method_name="PCA3")
    if X_umap3 is not None:
        plot_3d_interactive_per_fault(X_umap3, y_embed, variant, out_dir, method_name="UMAP3")
        
        
    cluster_df = compute_cluster_purity_metrics(X_pca3, y_embed, LABEL_COLS)
    cluster_df.to_csv(os.path.join(out_dir, "cluster_purity_metrics.csv"), index=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=cluster_df,
        x="purity",
        y="fault",
        palette="magma"
    )
    plt.title(f"{variant} — Cluster Purity Ranking")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_purity_ranking.png"), dpi=300)
    plt.close()
     
        
    # -----------------------------------------
    # CROSS VALIDATION
    # -----------------------------------------
    cv_dir = os.path.join(out_dir, "cross_validation")
    os.makedirs(cv_dir, exist_ok=True)

    if SESSION_AWARE:
        print(">>> Running SESSION-AWARE CV")
        cv_groups = sessions
    else:
        print(">>> Running RANDOM CV")
        cv_groups = None  # triggers normal KFold

    cross_validate_model("XGBoost", build_model_xgb, X.values, y, cv_groups, feature_names, cv_dir, variant)
    cross_validate_model("LightGBM", build_model_lgbm, X.values, y, cv_groups, feature_names, cv_dir, variant)
    cross_validate_model("RandomForest", build_model_rf, X.values, y, cv_groups, feature_names, cv_dir, variant)
    cross_validate_model("MLP", build_model_mlp, X.values, y, cv_groups, feature_names, cv_dir, variant)

    # -----------------------------------------
    # 2D EMBEDDINGS
    # -----------------------------------------
    plot_all_embeddings(X_embed, y_embed, feature_names, out_dir, variant)

    # -----------------------------------------
    # LOAD TRAINED MODELS FOR INTERPRETATION
    # -----------------------------------------
    saved_single = load_saved_models(variant, model_type="single")
    saved_multi  = load_saved_models(variant, model_type="multi")

    metrics_rows = []
    
    
    per_fault_preds = {}
    per_fault_macro_f1 = {}

    # -----------------------------------------
    # SINGLE-LABEL MODELS
    # -----------------------------------------
    for (fault, model_name), model in saved_single.items():
        print(f"\n--- {variant} — {model_name} — {fault} ---")

        fault_idx = LABEL_COLS.index(fault)
        y_test_fault = y_test[:, fault_idx]

        y_pred = model.predict(X_test_s)

        macro_f1 = f1_score(y_test_fault, y_pred, average="macro")
        acc = accuracy_score(y_test_fault, y_pred)
        
        per_fault_preds[fault] = y_pred
        per_fault_macro_f1[fault] = macro_f1

        metrics_rows.append({
            "variant": variant,
            "model": model_name,
            "fault": fault,
            "macro_f1": macro_f1,
            "accuracy": acc,
        })

        plot_confusion_matrices(
            y_test_fault.reshape(-1, 1),
            y_pred.reshape(-1, 1),
            [fault],
            out_dir,
            f"{model_name}_{fault}"
        )

        if any(x in model_name for x in ["XGBoost", "LightGBM", "RandomForest"]):
            plot_feature_importance_tree(model, feature_names, out_dir, f"{model_name}_{fault}")

            X_sample = X_test_s[:500] if X_test_s.shape[0] > 500 else X_test_s
            shap_summary_tree(model, X_sample, feature_names, out_dir, f"{model_name}_{fault}", fault=fault)

    # Convert dicts into aligned arrays
    macro_f1_array = np.array([per_fault_macro_f1[f] for f in LABEL_COLS])
    y_pred_matrix = np.column_stack([per_fault_preds[f] for f in LABEL_COLS])

    
    # -----------------------------------------
    # MULTI-LABEL MODELS
    # -----------------------------------------
    for model_name, model in saved_multi.items():
        print(f"\n--- {variant} — {model_name} (multi-label) ---")

        y_pred = model.predict(X_test_s)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)

        metrics_rows.append({
            "variant": variant,
            "model": model_name,
            "fault": "ALL",
            "macro_f1": macro_f1,
            "accuracy": acc,
        })

        plot_confusion_matrices(y_test, y_pred, LABEL_COLS, out_dir, model_name)

        if any(x in model_name for x in ["XGBoost", "LightGBM", "RandomForest"]):
            for fault in LABEL_COLS:
                fault_idx = LABEL_COLS.index(fault)
                est = model.estimators_[fault_idx]
                X_sample = X_test_s[:500] if X_test_s.shape[0] > 500 else X_test_s
                shap_summary_tree(est, X_sample, feature_names, out_dir, model_name, fault=fault)
    
    # Support counts
    support = compute_fault_support(y_embed)
    
    # Confusion difficulty (per fault)
    confusion_diff = compute_confusion_difficulty(y_test, y_pred_matrix)
    
    # Embedding difficulty (use PCA3 or PCA2)
    embed_diff = compute_embedding_difficulty(X_pca3, y_embed)
    
    # Co-occurrence difficulty (mean conditional co-occurrence)
    cooccur_diff = co_norm.mean(axis=1)

    
    difficulty_df = compute_fault_difficulty(
        macro_f1=macro_f1_array,  # from your model results
        confusion_diff=confusion_diff,
        support=support,
        cooccur_norm=cooccur_diff,
        embed_diff=embed_diff,
        labels=LABEL_COLS
    )
    
    difficulty_df.to_csv(os.path.join(out_dir, "fault_difficulty_ranking.csv"), index=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=difficulty_df,
        x="difficulty",
        y="fault",
        palette="viridis"
    )
    plt.title(f"{variant} — Fault Difficulty Ranking")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fault_difficulty_ranking.png"), dpi=300)
    plt.close()

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)


def run_all_variants():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for fname in os.listdir(ML_DATA_FOLDER):
        if fname.endswith(".csv"):
            path = os.path.join(ML_DATA_FOLDER, fname)
            run_for_variant(path)


if __name__ == "__main__":
    run_all_variants()
