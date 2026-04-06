import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,log_loss,confusion_matrix,roc_curve,auc)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from IPython.display import display

def evaluate(y_true, y_pred, y_prob, class_names=None, show=True):

    metrics = {}

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, average="macro",zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, average="macro",zero_division=0)
    metrics["F1_Score"] = f1_score(y_true, y_pred, average="macro",zero_division=0)
    metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    metrics["Log_loss"] = log_loss(y_true, y_prob)

    df = pd.DataFrame(metrics.items(), columns=["Metrics", "Value"])

    if show:
        print("\n===== METRICS =====")
        display(df.style.hide(axis="index"))
        print("\n============================ CONFUSION MATRIX =============================")
        plot_confusion(y_true, y_pred)
        print("\n==================================== ROC CURVE ====================================")
        plot_roc(y_true, y_prob)

    return metrics

def plot_confusion(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    idx = list(range(n_classes))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cm_cmap", ["#cce5ff", "#f7c948", "#e05c2a", "#1a7a2e"]
    )

    fig, ax = plt.subplots(figsize=(6.7,6))

    sns.heatmap(
        cm_norm,
        annot=False,
        cmap=cmap,
        linewidths=0.4,
        linecolor="white",
        xticklabels=idx,
        yticklabels=idx,
        vmin=0, vmax=1,
        ax=ax,
        cbar_kws={"shrink": 0.75},
    )

    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted", fontsize=9, labelpad=6)
    ax.set_ylabel("True", fontsize=9, labelpad=6)
    ax.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    plt.show()

def plot_roc(y_true, y_prob):

    n_classes = y_prob.shape[1]
    plt.figure(figsize=(8,8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

def compare_splits(train_m, val_m, test_m):

    df = pd.DataFrame([train_m, val_m, test_m],index=["train", "val", "test"])
    print("\n================ SPLIT COMPARISON ================")
    display(df.style.format("{:.4f}"))

    metrics = df.columns.tolist()
    x = np.arange(len(metrics))
    width = 0.25
    train_vals = df.loc["train"].values
    val_vals = df.loc["val"].values
    test_vals = df.loc["test"].values

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, train_vals, width, label="train")
    plt.bar(x, val_vals, width, label="val")
    plt.bar(x + width, test_vals, width, label="test")
    plt.xticks(x, metrics, rotation=0)
    plt.ylabel("Score")
    plt.title("Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df

def evaluate_model(model, loader, device, class_names=None, show=True):

    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    return evaluate(y_true, y_pred, y_prob, class_names, show)