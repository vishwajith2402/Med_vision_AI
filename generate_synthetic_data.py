from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_SIZE = 256


def add_scan_texture(rng: np.random.Generator) -> np.ndarray:
    image = rng.normal(95, 18, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)

    y_grid, x_grid = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    center = IMAGE_SIZE / 2
    distance = np.sqrt((x_grid - center) ** 2 + (y_grid - center) ** 2)
    vignette = np.clip(1.15 - distance / 240, 0.45, 1.0)
    image *= vignette

    for _ in range(8):
        x1 = int(rng.integers(20, IMAGE_SIZE - 20))
        y1 = int(rng.integers(20, IMAGE_SIZE - 20))
        x2 = int(np.clip(x1 + rng.normal(0, 45), 10, IMAGE_SIZE - 10))
        y2 = int(np.clip(y1 + rng.normal(0, 45), 10, IMAGE_SIZE - 10))
        color = int(rng.integers(95, 145))
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    return np.clip(image, 0, 255).astype(np.uint8)


def create_normal_image(rng: np.random.Generator) -> np.ndarray:
    image = add_scan_texture(rng)
    cv2.ellipse(image, (128, 132), (72, 96), 0, 0, 360, int(rng.integers(110, 145)), 2)
    cv2.ellipse(image, (92, 130), (28, 64), -8, 0, 360, int(rng.integers(75, 105)), -1)
    cv2.ellipse(image, (164, 130), (28, 64), 8, 0, 360, int(rng.integers(75, 105)), -1)
    cv2.line(image, (128, 50), (128, 218), int(rng.integers(130, 165)), 2, cv2.LINE_AA)
    return cv2.GaussianBlur(image, (5, 5), 0)


def create_disease_image(rng: np.random.Generator) -> np.ndarray:
    image = create_normal_image(rng)

    lesion_count = int(rng.integers(1, 4))
    for _ in range(lesion_count):
        x = int(rng.integers(70, 186))
        y = int(rng.integers(70, 190))
        radius = int(rng.integers(10, 28))
        intensity = int(rng.integers(165, 230))
        overlay = image.copy()
        cv2.circle(overlay, (x, y), radius, intensity, -1, cv2.LINE_AA)
        image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)

    for _ in range(2):
        x1 = int(rng.integers(45, 210))
        y1 = int(rng.integers(45, 210))
        x2 = int(np.clip(x1 + rng.normal(0, 55), 10, IMAGE_SIZE - 10))
        y2 = int(np.clip(y1 + rng.normal(0, 55), 10, IMAGE_SIZE - 10))
        cv2.line(image, (x1, y1), (x2, y2), int(rng.integers(170, 235)), 2, cv2.LINE_AA)

    return cv2.GaussianBlur(image, (3, 3), 0)


def generate_dataset(output_dir: Path, labels_file: Path, samples_per_class: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(1, samples_per_class + 1):
        normal_name = f"synthetic_normal_{index:03d}.png"
        disease_name = f"synthetic_disease_{index:03d}.png"

        normal_image = create_normal_image(rng)
        disease_image = create_disease_image(rng)

        cv2.imwrite(str(output_dir / normal_name), normal_image)
        cv2.imwrite(str(output_dir / disease_name), disease_image)

        rows.append({"filename": normal_name, "label": "Normal"})
        rows.append({"filename": disease_name, "label": "Disease"})

    labels = pd.DataFrame(rows)
    labels.to_csv(labels_file, index=False)

    print(f"Generated {len(rows)} synthetic images in {output_dir}")
    print(f"Wrote labels to {labels_file}")
    print("This dataset is for pipeline testing only, not medical accuracy.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic demo images for MedAI Vision.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--labels", type=Path, default=Path("Labels.csv"))
    parser.add_argument("--samples-per-class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        labels_file=args.labels,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
