from pathlib import Path
from collections import Counter
import json

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BinaryPetDataset(Dataset):
    """Wrap Oxford-IIIT Pet and map breed labels to Cat=0, Dog=1."""

    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, breed_label = self.base_dataset[self.indices[item]]
        breed_name = self.base_dataset.classes[int(breed_label)]
        binary_label = 0 if breed_name in CAT_BREEDS else 1
        if self.transform is not None:
            image = self.transform(image)
        return image, binary_label


def make_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def load_base_dataset(data_dir="data", download=True):
    return OxfordIIITPet(
        root=str(data_dir),
        split="trainval",
        target_types="category",
        download=download,
    )


def binary_labels_for_dataset(base_dataset):
    """Read labels once to create a stratified split."""
    labels = []
    for i in range(len(base_dataset)):
        _, breed_label = base_dataset[i]
        breed_name = base_dataset.classes[int(breed_label)]
        labels.append(0 if breed_name in CAT_BREEDS else 1)
    return np.asarray(labels, dtype=np.int64)


def make_splits(base_dataset, seed=42):
    labels = binary_labels_for_dataset(base_dataset)
    all_indices = np.arange(len(labels))
    train_idx, remainder_idx = train_test_split(
        all_indices, test_size=0.30, random_state=seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        remainder_idx, test_size=0.50, random_state=seed, stratify=labels[remainder_idx]
    )
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
        "seed": int(seed),
        "label_map": {"0": "Cat", "1": "Dog"},
        "class_counts": {
            "train": dict(Counter(str(int(labels[i])) for i in train_idx)),
            "val": dict(Counter(str(int(labels[i])) for i in val_idx)),
            "test": dict(Counter(str(int(labels[i])) for i in test_idx)),
        },
    }


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
