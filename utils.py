"""
utils.py
Data loading, preprocessing, and visualisation helpers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_image_paths_and_labels(dataset_dir: str):
    """
    Walk a directory structured as:
        dataset_dir/
            class_name_1/img1.jpg ...
            class_name_2/img1.jpg ...

    Returns
    -------
    image_paths : list of str
    labels      : list of str (class names)
    class_names : sorted list of unique class names
    """
    image_paths, labels = [], []
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    for cls in class_names:
        cls_dir = os.path.join(dataset_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)

    return image_paths, labels, class_names


def encode_labels(labels: list):
    """
    Encode string class labels to integer indices.

    Returns
    -------
    y       : np.ndarray of int
    encoder : fitted LabelEncoder
    """
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    return y, le


def stratified_split(image_paths, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Stratified split into train / val / test sets.
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        image_paths, y, test_size=(test_size + val_size),
        stratify=y, random_state=random_state
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - val_ratio),
        stratify=y_tmp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(labels: list, class_names: list, title="Class Distribution"):
    from collections import Counter
    counts = Counter(labels)
    counts_ordered = [counts.get(c, 0) for c in class_names]

    plt.figure(figsize=(12, 4))
    plt.bar(class_names, counts_ordered, color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Image Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                           figsize=(12, 10), title="Confusion Matrix"):
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(clf, feature_names: list = None, top_n: int = 20):
    """
    Plot XGBoost feature importances (top_n features).
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    labels = (
        [feature_names[i] for i in indices]
        if feature_names else [f"Feature {i}" for i in indices]
    )

    plt.figure(figsize=(10, 5))
    plt.bar(range(top_n), importances[indices], color="coral")
    plt.xticks(range(top_n), labels, rotation=45, ha="right")
    plt.ylabel("Importance Score")
    plt.title(f"Top {top_n} XGBoost Feature Importances")
    plt.tight_layout()
    plt.show()
