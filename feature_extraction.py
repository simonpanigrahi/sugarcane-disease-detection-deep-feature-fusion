"""
feature_extraction.py
Handcrafted feature extraction: GLCM texture descriptors + HSV colour histograms.
Produces an 11-dimensional feature vector per image.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# ─────────────────────────────────────────────────────────────────────────────
# GLCM Texture Features (8 features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_glcm_features(image_gray: np.ndarray) -> np.ndarray:
    """
    Extract GLCM texture descriptors from a grayscale image.

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale image, shape (H, W), dtype uint8.

    Returns
    -------
    np.ndarray
        8-dimensional feature vector:
        [contrast, dissimilarity, homogeneity, energy, correlation, ASM,
         mean_gray, std_gray]
    """
    # Reduce bit depth for GLCM computation (256 → 32 levels)
    image_8bit = (image_gray / 8).astype(np.uint8)

    glcm = graycomatrix(
        image_8bit,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )

    contrast     = graycoprops(glcm, "contrast").mean()
    dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    homogeneity  = graycoprops(glcm, "homogeneity").mean()
    energy       = graycoprops(glcm, "energy").mean()
    correlation  = graycoprops(glcm, "correlation").mean()
    asm          = graycoprops(glcm, "ASM").mean()

    mean_gray = image_gray.mean() / 255.0
    std_gray  = image_gray.std()  / 255.0

    return np.array([
        contrast, dissimilarity, homogeneity, energy,
        correlation, asm, mean_gray, std_gray
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# HSV Colour Features (3 features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_hsv_features(image_bgr: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Extract normalised HSV histogram statistics.

    Parameters
    ----------
    image_bgr : np.ndarray
        BGR image (as returned by cv2.imread), shape (H, W, 3).
    bins : int
        Number of histogram bins per channel.

    Returns
    -------
    np.ndarray
        3-dimensional feature vector: [h_mean, s_mean, v_mean]
        (mean of per-channel normalised histogram).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()

    return np.array([h_hist.mean(), s_hist.mean(), v_hist.mean()], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Combined Handcrafted Feature Vector (11-dim)
# ─────────────────────────────────────────────────────────────────────────────

def extract_handcrafted_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extract the full 11-dimensional handcrafted feature vector.

    Parameters
    ----------
    image_bgr : np.ndarray
        BGR image of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        11-dimensional vector: GLCM (8) + HSV (3).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    glcm_feats = extract_glcm_features(gray)
    hsv_feats  = extract_hsv_features(image_bgr)
    return np.concatenate([glcm_feats, hsv_feats])


def batch_extract_handcrafted(image_paths: list, target_size=(224, 224)) -> np.ndarray:
    """
    Extract handcrafted features from a list of image file paths.

    Parameters
    ----------
    image_paths : list of str
    target_size : tuple (W, H)

    Returns
    -------
    np.ndarray, shape (N, 11)
    """
    features = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.resize(img, target_size)
        features.append(extract_handcrafted_features(img))
    return np.array(features, dtype=np.float32)
