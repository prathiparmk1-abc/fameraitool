#!/usr/bin/env python3
"""classify.py - leaf photo -> JSON for the advisory pipeline.

Prototype for the Arm Bharat SoC Challenge (Phase 1).

Usage:
    python classify.py --list-labels
    python classify.py test_images/leaf1.jpg test_images/leaf2.jpg
    python classify.py test_images/*.jpg --region Karnataka --language hi

What it does:
  1. Runs a pretrained MobileNetV2 (fine-tuned on PlantVillage, 38 classes).
  2. Splits the label into crop + problem.
  3. Estimates severity with a simple colour heuristic (OpenCV).
  4. Prints one JSON object per image (the interface agreed with the Llama side).

Why the image preprocessing is written by hand:
  The model's preprocessor_config.json names an old class
  ("MobileNetV2FeatureExtractor") that recent transformers versions no longer
  recognise. The steps below copy that file exactly: resize the shortest edge
  to 256, centre-crop 224x224, scale to 0-1, then normalise with mean 0.5 and
  std 0.5. This also removes a dependency on transformers' image processors.

Limitations (say these in the report):
  - PlantVillage photos are taken on plain backgrounds; accuracy on real
    field photos will be lower.
  - The severity rule is a crude prototype heuristic, not validated.
  - Low-confidence predictions are flagged so the app can refuse to advise.
  - The model's licence is listed as "other"; check it before submission.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
LOW_CONFIDENCE = 0.60  # below this, the app should not give disease advice

# Copied from the model's preprocessor_config.json
SHORT_EDGE = 256
CROP_SIZE = 224
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5


def slug(text):
    """'Early blight' -> 'early_blight'"""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# Labels that do not follow the 'X with Y' / 'Healthy X Plant' pattern.
LABEL_OVERRIDES = {
    "Apple Scab": ("apple", "scab"),
    "Cedar Apple Rust": ("apple", "cedar_rust"),
    "Tomato Yellow Leaf Curl Virus": ("tomato", "yellow_leaf_curl_virus"),
    "Tomato Mosaic Virus": ("tomato", "mosaic_virus"),
    "Tomato with Spider Mites or Two-spotted Spider Mite": ("tomato", "spider_mites"),
}


def parse_label(label):
    """Split a class label into (crop, problem) slugs.

    Handles the styles seen in this model's 38 labels:
      'Tomato with Late Blight'  -> ('tomato', 'late_blight')
      'Healthy Tomato Plant'     -> ('tomato', 'healthy')
      'Tomato Mosaic Virus'      -> ('tomato', 'mosaic_virus')   (override)
      'Tomato___Early_blight'    -> ('tomato', 'early_blight')   (older style)
    Anything else gives crop 'unknown' (check with --list-labels).
    """
    text = label.strip()
    if text in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[text]
    m = re.match(r"^healthy\s+(.+?)(?:\s+plant)?$", text, flags=re.IGNORECASE)
    if m:
        return slug(m.group(1)), "healthy"
    m = re.match(r"^(.+?)\s+with\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        return slug(m.group(1)), slug(m.group(2))
    if "___" in text:
        crop, problem = text.split("___", 1)
        return slug(crop), slug(problem)
    return "unknown", slug(text)


def preprocess(image):
    """PIL image -> float32 array of shape (1, 3, 224, 224)."""
    w, h = image.size
    if w <= h:
        new_w, new_h = SHORT_EDGE, int(SHORT_EDGE * h / w)
    else:
        new_w, new_h = int(SHORT_EDGE * w / h), SHORT_EDGE
    image = image.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - CROP_SIZE) // 2
    top = (new_h - CROP_SIZE) // 2
    image = image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    return arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def load_model():
    """Load the classifier (torch and transformers are imported only here)."""
    import torch
    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return model, torch


def predict(model, torch, image, top_k):
    """Return the top-k [{'label', 'score'}] for one PIL image."""
    x = torch.from_numpy(preprocess(image))
    with torch.no_grad():
        logits = model(pixel_values=x).logits
    probs = torch.softmax(logits, dim=-1)[0]
    k = min(top_k, probs.numel())
    scores, indices = torch.topk(probs, k)
    id2label = model.config.id2label
    return [
        {"label": id2label.get(int(i), str(int(i))), "score": float(s)}
        for s, i in zip(scores, indices)
    ]


def estimate_severity(image_path, is_healthy):
    """Crude severity: share of leaf pixels that are not green.

    Returns (severity, fraction). Prototype heuristic only.
    """
    if is_healthy:
        return "none", 0.0
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return "unknown", None
    bgr = cv2.resize(bgr, (256, 256))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    leaf = (s > 40) & (v > 25)          # coloured pixels, not grey/black background
    if int(leaf.sum()) < 500:
        return "unknown", None
    green = (h >= 35) & (h <= 85)       # OpenCV hue range is 0-179
    lesion = leaf & ~green
    fraction = float(lesion.sum()) / float(leaf.sum())
    if fraction < 0.10:
        severity = "mild"
    elif fraction < 0.30:
        severity = "moderate"
    else:
        severity = "severe"
    return severity, round(fraction, 3)


def classify_image(model, torch, path, top_k, region, language):
    image = Image.open(path).convert("RGB")
    results = predict(model, torch, image, top_k)
    top = results[0]
    crop, problem = parse_label(top["label"])
    is_healthy = problem == "healthy"
    severity, fraction = estimate_severity(path, is_healthy)
    confidence = round(float(top["score"]), 4)
    return {
        "file": str(path),
        "crop": crop,
        "problem": problem,
        "confidence": confidence,
        "severity": severity,
        "damaged_area_fraction": fraction,
        "region": region,
        "language": language,
        "status": "ok" if confidence >= LOW_CONFIDENCE else "low_confidence",
        "top_k": [
            {"label": r["label"], "score": round(float(r["score"]), 4)} for r in results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Leaf photo -> JSON")
    parser.add_argument("images", nargs="*", help="image files to classify")
    parser.add_argument("--list-labels", action="store_true",
                        help="print all class labels and exit")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--region", default="Karnataka")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    model, torch = load_model()

    if args.list_labels:
        for idx, label in sorted(model.config.id2label.items()):
            crop, problem = parse_label(label)
            flag = "   <-- CHECK" if crop == "unknown" else ""
            print(f"{idx:>2}  {label}  ->  crop={crop}, problem={problem}{flag}")
        return 0

    if not args.images:
        parser.error("give at least one image path (or use --list-labels)")

    exit_code = 0
    for name in args.images:
        path = Path(name)
        if not path.is_file():
            print(f"SKIPPED (not a file): {path}", file=sys.stderr)
            exit_code = 1
            continue
        result = classify_image(model, torch, path, args.top_k, args.region, args.language)
        print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
