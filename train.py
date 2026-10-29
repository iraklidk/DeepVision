import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import (
    load_base_dataset, make_splits, make_transforms, BinaryPetDataset, save_json
)
from src.model import build_model, set_training_modes


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, training, unfreeze_last_block):
    if training:
        set_training_modes(model, unfreeze_last_block)
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Cat vs Dog ResNet-18.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--unfreeze-last-block", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Loading Oxford-IIIT Pet (trainval split). The first run downloads the dataset.")
    base_dataset = load_base_dataset("data", download=True)
    split_data = make_splits(base_dataset, seed=args.seed)
    save_json(split_data, output_dir / "splits.json")

    train_transform, eval_transform = make_transforms()
    train_dataset = BinaryPetDataset(base_dataset, split_data["train"], train_transform)
    val_dataset = BinaryPetDataset(base_dataset, split_data["val"], eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model, weights = build_model(args.unfreeze_last_block)
    model = model.to(device)

    # Inverse-frequency weights help account for more dog than cat images.
    counts_by_label = split_data["class_counts"]["train"]
    counts = np.array(
        [int(counts_by_label.get(str(label), 0)) for label in (0, 1)],
        dtype=np.int64,
    )
    class_weights = len(split_data["train"]) / (2.0 * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)

    history = []
    best_val_loss = float("inf")
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, True,
            args.unfreeze_last_block
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, False,
            args.unfreeze_last_block
        )
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f} | "
            f"elapsed {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": ["Cat", "Dog"],
                "unfreeze_last_block": args.unfreeze_last_block,
                "best_val_loss": best_val_loss,
                "seed": args.seed,
                "architecture": "ResNet-18",
                "weights": "ResNet18_Weights.DEFAULT",
            }, output_dir / "best_model.pth")

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Model checkpoint: {output_dir / 'best_model.pth'}")
    print("Next run: python evaluate.py --run-name " + args.run_name)


if __name__ == "__main__":
    main()
