from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np


IMAGE_SIZE = (128, 128)
MODEL_PATH = Path("models/medai_random_forest.joblib")
CLASS_NAMES = {
    0: "Normal",
    1: "Disease",
}


def read_image(image_source: str | Path | bytes) -> np.ndarray:
    """Read an image from a file path or uploaded bytes."""
    if isinstance(image_source, bytes):
        image_array = np.frombuffer(image_source, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(image_source), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to read image. Please use a valid JPG or PNG file.")

    return image


def preprocess_image(image: np.ndarray, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    equalized = cv2.equalizeHist(denoised)
    return equalized


def extract_features(image: np.ndarray) -> np.ndarray:
    processed = preprocess_image(image)

    pixel_features = (processed.astype(np.float32) / 255.0).flatten()

    histogram = cv2.calcHist([processed], [0], None, [32], [0, 256]).flatten()
    histogram = histogram / (histogram.sum() + 1e-7)

    edges = cv2.Canny(processed, 80, 160)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)

    moments = cv2.HuMoments(cv2.moments(processed)).flatten()
    moments = -np.sign(moments) * np.log10(np.abs(moments) + 1e-7)

    features = np.concatenate(
        [
            pixel_features,
            histogram.astype(np.float32),
            edge_density,
            moments.astype(np.float32),
        ]
    )
    return features.astype(np.float32)


def load_model(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train the model with: python train_model.py"
        )
    return joblib.load(path)


def predict_image(image_source: str | Path | bytes, model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    bundle = load_model(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    image = read_image(image_source)
    features = extract_features(image).reshape(1, -1)

    prediction_index = int(model.predict(features)[0])
    predicted_label = label_encoder.inverse_transform([prediction_index])[0]

    confidence = None
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba(features)[0]
        for class_index, probability in zip(model.classes_, probability_values):
            label = label_encoder.inverse_transform([int(class_index)])[0]
            probabilities[str(label)] = float(probability)
        confidence = float(max(probability_values))

    return {
        "label": str(predicted_label),
        "confidence": confidence,
        "probabilities": probabilities,
    }
