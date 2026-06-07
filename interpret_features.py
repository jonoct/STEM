# interpret_features.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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
    
from sklearn.model_selection import GroupKFold

ML_DATA_FOLDER = "./ml_data"
RESULTS_FOLDER = "./results/interpretability"

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


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop(columns=LABEL_COLS + ["session_id", "rep_start", "rep_end"])
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
    # model is OneVsRest with tree base estimators
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
        plt.xlabel("Importance")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{model_name}_featimp_{label}.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def shap_summary_tree(model, X_sample, feature_names, out_dir, model_name, fault="knee_valgus"):
    if model_name == "MLP":
        return  # skip non-tree models

    idx_fault = LABEL_COLS.index(fault)
    est = model.estimators_[idx_fault]

    # Skip constant predictors (no variability in training)
    from sklearn.dummy import DummyClassifier
    if isinstance(est, DummyClassifier):
        print(f"[SHAP] Skipping {model_name} / {fault} — constant predictor.")
        return

    # Skip unsupported models
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


def permutation_importance_mlp(model, X_test_s, y_test, feature_names, out_dir, model_name, top_n=20):
    print(f"[MLP] Computing permutation importance...")

    # scoring = "f1_macro" works for multilabel only if averaged manually
    # so we use accuracy as a stable metric
    result = permutation_importance(
        model,
        X_test_s,
        y_test,
        scoring="accuracy",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.perm_importances_ = result.importances_mean  # <-- store for later
    importances = result.importances_mean
    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(6, 5))
    sns.barplot(
        x=importances[idx],
        y=[feature_names[j] for j in idx],
        orient="h"
    )
    plt.title(f"{model_name} — Permutation Importance (Top {top_n})")
    plt.xlabel("Importance (mean accuracy drop)")
    plt.tight_layout()

    fname = os.path.join(out_dir, f"{model_name}_perm_importance.png")
    plt.savefig(fname, dpi=200)
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


def cross_validate_model(model_name, model_builder, X, y, sessions, feature_names, out_dir, variant, n_splits=5):
    print(f"[CV] Running {n_splits}-fold session-aware CV for {model_name}...")

    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=sessions)):
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


def run_for_variant(dataset_path):
    base = os.path.basename(dataset_path)
    variant = base[len("squat_ml_dataset_"):-len(".csv")]
    print(f"\n=== INTERPRETABILITY: {variant} ===")

    out_dir = os.path.join(RESULTS_FOLDER, variant)
    os.makedirs(out_dir, exist_ok=True)

    X, y, sessions = load_dataset(dataset_path)
    feature_names = X.columns

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_sess, val_sess, test_sess
    ) = session_aware_split(X, y, sessions)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Embeddings on the full test set (or combined train+test if you prefer)
    X_embed = np.vstack([X_train_s, X_val_s, X_test_s])
    y_embed = np.vstack([y_train, y_val, y_test])
    
    # -------------------------
    # 5-FOLD CROSS VALIDATION
    # -------------------------
    cv_dir = os.path.join(out_dir, "cross_validation")
    os.makedirs(cv_dir, exist_ok=True)
    
    cross_validate_model(
        "XGBoost",
        build_model_xgb,
        X.values,
        y,
        sessions,
        feature_names,
        cv_dir,
        variant
    )
    
    cross_validate_model(
        "LightGBM",
        build_model_lgbm,
        X.values,
        y,
        sessions,
        feature_names,
        cv_dir,
        variant
    )
    
    cross_validate_model(
        "RandomForest",
        build_model_rf,
        X.values,
        y,
        sessions,
        feature_names,
        cv_dir,
        variant
    )
    
    cross_validate_model(
        "MLP",
        build_model_mlp,
        X.values,
        y,
        sessions,
        feature_names,
        cv_dir,
        variant
    )

    
    # Choose which fault to color by (can change per run)
    # plot_embeddings(
    #     X_embed,
    #     y_embed,
    #     feature_names,
    #     out_dir,
    #     variant,
    #     fault="knee_valgus",  # or "good_technique", etc.
    # )
    
    plot_all_embeddings(
        X_embed,
        y_embed,
        feature_names,
        out_dir,
        variant
    )


    models = build_models()

    metrics_rows = []

    for model_name, model in models:
        print(f"\n--- {variant} — {model_name} ---")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)

        print("Macro F1:", macro_f1)
        print("Accuracy:", acc)
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=LABEL_COLS))

        metrics_rows.append({
            "variant": variant,
            "model": model_name,
            "macro_f1": macro_f1,
            "accuracy": acc,
        })

        # confusion matrices (normalised)
        plot_confusion_matrices(y_test, y_pred, LABEL_COLS, out_dir, model_name)

        # feature importance + SHAP for tree models
        if model_name in ["XGBoost", "LightGBM", "RandomForest"]:
            plot_feature_importance_tree(model, feature_names, out_dir, model_name)

            # SHAP on a sample of test data
            X_sample = X_test_s
            if X_sample.shape[0] > 500:
                idx = np.random.choice(X_sample.shape[0], 500, replace=False)
                X_sample = X_sample[idx]
            shap_summary_tree(model, X_sample, feature_names, out_dir, model_name)
        
        # permutation importance for MLP
        if model_name == "MLP":
            permutation_importance_mlp(
                model,
                X_test_s,
                y_test,
                feature_names,
                out_dir,
                model_name
            )
        
        # After training each model
        top_features = get_top_k_features(model_name, model, feature_names, k=10)
        
        if len(top_features) > 0:
            plot_feature_distributions(
                X_embed,      # combined train+val+test
                y_embed,
                feature_names,
                out_dir,
                variant,
                fault="knee_valgus",  # or any fault you want
                top_features=top_features
            )

    # save metrics table for this variant
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)


def run_all_variants():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for fname in os.listdir(ML_DATA_FOLDER):
        if fname.startswith("squat_ml_dataset_") and fname.endswith(".csv"):
            path = os.path.join(ML_DATA_FOLDER, fname)
            run_for_variant(path)


if __name__ == "__main__":
    run_all_variants()
