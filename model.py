"""
model.py
MobileNetV2 deep feature extraction and XGBoost classifier training.
Produces 1,291-dim fused feature vectors (1,280 deep + 11 handcrafted).
"""

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
# MobileNetV2 Deep Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_extractor(input_shape=(224, 224, 3)) -> Model:
    """
    Build a MobileNetV2 feature extraction model (no top layer).

    Returns
    -------
    keras.Model
        Outputs 1,280-dimensional global average pooled embeddings.
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",   # global average pooling → (batch, 1280)
    )
    base.trainable = False
    return base


def extract_deep_features(model: Model, image_paths: list,
                           batch_size: int = 32, target_size=(224, 224)) -> np.ndarray:
    """
    Extract MobileNetV2 embeddings from a list of image paths.

    Parameters
    ----------
    model       : Keras Model (output: 1,280-dim)
    image_paths : list of str
    batch_size  : int
    target_size : tuple (H, W)

    Returns
    -------
    np.ndarray, shape (N, 1280)
    """
    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        batch_imgs  = []

        for p in batch_paths:
            img = keras_image.load_img(p, target_size=target_size)
            arr = keras_image.img_to_array(img)
            arr = preprocess_input(arr)
            batch_imgs.append(arr)

        batch_tensor = np.array(batch_imgs)
        feats = model.predict(batch_tensor, verbose=0)
        all_features.append(feats)

    return np.vstack(all_features)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Fusion
# ─────────────────────────────────────────────────────────────────────────────

def fuse_features(deep_features: np.ndarray, handcrafted_features: np.ndarray) -> np.ndarray:
    """
    Concatenate deep (1,280) and handcrafted (11) feature matrices.

    Parameters
    ----------
    deep_features        : np.ndarray, shape (N, 1280)
    handcrafted_features : np.ndarray, shape (N, 11)

    Returns
    -------
    np.ndarray, shape (N, 1291)
    """
    assert deep_features.shape[0] == handcrafted_features.shape[0], \
        "Mismatch: deep and handcrafted feature matrices have different number of samples."
    return np.concatenate([deep_features, handcrafted_features], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  n_estimators: int = 300,
                  max_depth: int = 6,
                  learning_rate: float = 0.1,
                  random_state: int = 42) -> XGBClassifier:
    """
    Train an XGBoost classifier on fused feature vectors.

    Returns
    -------
    XGBClassifier (fitted)
    """
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf: XGBClassifier, X_test: np.ndarray,
                   y_test: np.ndarray, class_names: list = None):
    """
    Print accuracy, classification report, and return confusion matrix.
    """
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    return cm


def save_model(clf: XGBClassifier, path: str = "xgboost_model.pkl"):
    joblib.dump(clf, path)
    print(f"Model saved to {path}")


def load_model(path: str = "xgboost_model.pkl") -> XGBClassifier:
    return joblib.load(path)
