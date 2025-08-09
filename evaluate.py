import os
import time
from typing import Dict, List, Tuple

from image_classifier import ImageClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root_dir: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    if not os.path.isdir(root_dir):
        return items
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            _, ext = os.path.splitext(fname)
            if ext.lower() in IMAGE_EXTENSIONS:
                items.append((os.path.join(class_dir, fname), class_name))
    return items


def evaluate(val_dir: str = "data/val", model_path: str = "image_classification_model.pth",
             threshold: float = 0.60, use_tta: bool = False) -> Dict[str, float]:
    classifier = ImageClassifier(model_path=model_path)

    samples = iter_images(val_dir)
    if not samples:
        print(f"No validation images found under {val_dir}.")
        return {}

    total = 0
    correct = 0
    uncertain = 0
    per_class_total: Dict[str, int] = {}
    per_class_correct: Dict[str, int] = {}
    t0 = time.time()

    for image_path, true_label in samples:
        total += 1
        per_class_total[true_label] = per_class_total.get(true_label, 0) + 1

        result = classifier.classify_image(image_path, confidence_threshold=threshold, use_tta=use_tta)
        if result is None:
            continue
        pred = result.get("class")
        if pred == "uncertain":
            uncertain += 1
        if pred == true_label:
            correct += 1
            per_class_correct[true_label] = per_class_correct.get(true_label, 0) + 1

    elapsed = time.time() - t0
    avg_ms = (elapsed / max(total, 1)) * 1000.0

    overall_acc = correct / total if total else 0.0
    print({
        "total": total,
        "correct": correct,
        "uncertain": uncertain,
        "accuracy": round(overall_acc * 100.0, 2),
        "avg_inference_ms": round(avg_ms, 2),
        "per_class": {
            cls: round((per_class_correct.get(cls, 0) / per_class_total.get(cls, 1)) * 100.0, 2)
            for cls in sorted(per_class_total.keys())
        },
    })
    return {
        "total": total,
        "correct": correct,
        "uncertain": uncertain,
        "accuracy": overall_acc,
        "avg_inference_ms": avg_ms,
    }


if __name__ == "__main__":
    evaluate()


