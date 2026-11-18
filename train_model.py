from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model_utils import MODEL_PATH, extract_features, read_image


VALID_LABELS = {"Normal", "Disease"}


def load_dataset(labels_file: Path, images_dir: Path) -> tuple[list[Path], list[str]]:
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    labels = pd.read_csv(labels_file)
    required_columns = {"filename", "label"}
    missing_columns = required_columns.difference(labels.columns)
    if missing_columns:
        raise ValueError(
            f"Labels file must contain columns: {', '.join(sorted(required_columns))}. "
            f"Missing: {', '.join(sorted(missing_columns))}"
        )

    image_paths: list[Path] = []
    target_labels: list[str] = []

    for row in labels.itertuples(index=False):
        filename = str(row.filename).strip()
        label = str(row.label).strip()
        image_path = images_dir / filename

        if not filename:
            continue
        if label not in VALID_LABELS:
            raise ValueError(
                f"Invalid label '{label}' for {filename}. Use only: Normal or Disease."
            )
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in labels file was not found: {image_path}")

        image_paths.append(image_path)
        target_labels.append(label)

    if len(image_paths) < 4:
        raise ValueError("At least 4 labeled images are required to train and evaluate the model.")

    if len(set(target_labels)) < 2:
        raise ValueError("Training requires both classes: Normal and Disease.")

    return image_paths, target_labels


def build_feature_matrix(image_paths: list[Path]) -> list:
    features = []
    for image_path in image_paths:
        image = read_image(image_path)
        features.append(extract_features(image))
    return features


def train_model(
    labels_file: Path,
    images_dir: Path,
    model_path: Path,
    test_size: float,
    random_state: int,
) -> None:
    image_paths, labels = load_dataset(labels_file, images_dir)
    features = build_feature_matrix(image_paths)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    stratify = encoded_labels if min(pd.Series(encoded_labels).value_counts()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "image_size": (128, 128),
            "labels_file": str(labels_file),
            "images_dir": str(images_dir),
        },
        model_path,
    )

    print(f"Model saved to: {model_path}")
    print(f"Test accuracy: {accuracy:.2%}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MedAI Vision Random Forest model.")
    parser.add_argument("--labels", type=Path, default=Path("Labels.csv"))
    parser.add_argument("--images", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        labels_file=args.labels,
        images_dir=args.images,
        model_path=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )
