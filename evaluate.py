import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)

from src.data import load_base_dataset, load_json, make_transforms, BinaryPetDataset, save_json
from src.model import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Cat vs Dog model.")
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path("outputs") / args.run_name
    checkpoint_path = run_dir / "best_model.pth"
    splits_path = run_dir / "splits.json"
    if not checkpoint_path.exists() or not splits_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint or split file in {run_dir}. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, _ = build_model(checkpoint.get("unfreeze_last_block", False))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    base_dataset = load_base_dataset("data", download=True)
    split_data = load_json(splits_path)
    _, eval_transform = make_transforms()
    test_dataset = BinaryPetDataset(base_dataset, split_data["test"], eval_transform)
    loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(predictions.tolist())
            y_true.extend(labels.numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy().tolist())

    labels_order = [0, 1]
    names = ["Cat", "Dog"]
    report = classification_report(
        y_true, y_pred, labels=labels_order, target_names=names,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    accuracy = accuracy_score(y_true, y_pred)
    summary = {
        "run_name": args.run_name,
        "test_accuracy": float(accuracy),
        "cat_precision": float(report["Cat"]["precision"]),
        "cat_recall": float(report["Cat"]["recall"]),
        "cat_f1": float(report["Cat"]["f1-score"]),
        "dog_precision": float(report["Dog"]["precision"]),
        "dog_recall": float(report["Dog"]["recall"]),
        "dog_f1": float(report["Dog"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix_labels": names,
        "confusion_matrix_rows_true_columns_predicted": cm.tolist(),
    }
    save_json(summary, run_dir / "test_metrics.json")
    save_json(report, run_dir / "classification_report.json")
    pd.DataFrame(
        cm, index=["true_cat", "true_dog"], columns=["pred_cat", "pred_dog"]
    ).to_csv(run_dir / "confusion_matrix.csv")
    pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred,
        "cat_probability": np.asarray(y_prob)[:, 0],
        "dog_probability": np.asarray(y_prob)[:, 1],
    }).to_csv(run_dir / "test_predictions.csv", index=False)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=labels_order, target_names=names, zero_division=0
    ))
    print(f"Saved metrics to {run_dir}")


if __name__ == "__main__":
    main()
