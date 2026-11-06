"""Dataset, splits and transforms (Phase 1).

Project rules enforced here:
  * The official KolektorSDD2 train/test split is always used.
  * Validation is a stratified slice of the official TRAIN. The test set is never used to
    choose hyperparameters or thresholds.
  * The defect mask is NOT a model input: `InspectDataset` returns (image, label) only.
    Masks are loaded separately with `load_mask`, to evaluate localization.
  * Augmentation only in train.

Expected layout: <root>/{train,test}/<id>.png and <id>_GT.png. A part is defective (1) when its
mask has any pixel > 0; the dataset ships no other labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

SEED = 42
MASK_SUFFIX = "_GT"
SPLIT_NAMES = ("train", "val", "test")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The images are VERTICAL: about 228 px wide x 635 px tall (PIL's .size is (width, height)).
# The project plan calls this "224 x 608" (wide x tall, multiples of 32). torchvision's
# Resize expects (height, width), hence (608, 224). Swapping it squashes and stretches the
# image (bug fixed on 2026-11-05, see docs/DECISIONES.md D-020).
IMG_SIZE = (608, 224)  # (height, width)


# --------------------------------------------------------------------------- #
# Index and splits
# --------------------------------------------------------------------------- #
def build_index(root: str | Path) -> pd.DataFrame:
    """One row per image of the official train/test folders.

    Columns: split, image, mask (paths relative to `root`, POSIX style), label (0 good /
    1 defective), defect_px (mask pixels > 0), width, height.
    Files whose stem is not purely numeric (e.g. "10301 (copy).png") are skipped: they are
    accidental duplicates and would leak between splits. A missing mask raises an error
    instead of silently labelling the part as good.
    """
    root = Path(root)
    rows, skipped = [], []
    for split in ("train", "test"):
        folder = root / split
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} does not exist. Is the dataset path correct?")
        for img in sorted(folder.glob("*.png")):
            if img.stem.endswith(MASK_SUFFIX):
                continue
            if not img.stem.isdigit():
                skipped.append(img.name)
                continue
            mask = img.with_name(f"{img.stem}{MASK_SUFFIX}.png")
            if not mask.exists():
                raise FileNotFoundError(f"Missing mask for {img}")
            m = np.array(Image.open(mask).convert("L"))
            with Image.open(img) as im:
                width, height = im.size
            rows.append(dict(
                split=split, image=img.relative_to(root).as_posix(),
                mask=mask.relative_to(root).as_posix(), label=int(m.max() > 0),
                defect_px=int((m > 0).sum()), width=width, height=height,
            ))
    if skipped:
        print(f"Warning: skipped {len(skipped)} non-numeric file(s): {skipped[:4]}")
    return pd.DataFrame(rows)


def make_splits(df: pd.DataFrame, val_frac: float = 0.2, seed: int = SEED):
    """Return (train, val, test) DataFrames.

    test = official 'test' split (untouched). val = `val_frac` of the official 'train',
    stratified by label, so train and val keep the same defect rate.
    """
    official_train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    train, val = train_test_split(
        official_train, test_size=val_frac, stratify=official_train["label"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    """Write train.csv / val.csv / test.csv so the whole team uses exactly the same splits."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        splits[name].to_csv(out_dir / f"{name}.csv", index=False)


def load_splits(split_dir: str | Path) -> dict[str, pd.DataFrame]:
    split_dir = Path(split_dir)
    return {name: pd.read_csv(split_dir / f"{name}.csv") for name in SPLIT_NAMES}


def check_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Leakage guard: no image may appear in two splits."""
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for img in splits[name]["image"]:
            if img in seen:
                raise AssertionError(f"{img} is in both '{seen[img]}' and '{name}'")
            seen[img] = name


def get_splits(root: str | Path, split_dir: str | Path, val_frac: float = 0.2,
               seed: int = SEED, rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """Load the saved CSV splits, or build and save them the first time."""
    split_dir = Path(split_dir)
    if not rebuild and all((split_dir / f"{n}.csv").exists() for n in SPLIT_NAMES):
        splits = load_splits(split_dir)
    else:
        train, val, test = make_splits(build_index(root), val_frac, seed)
        splits = {"train": train, "val": val, "test": test}
        save_splits(splits, split_dir)
    check_splits(splits)
    return splits


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_transforms(train: bool, size: tuple[int, int] = IMG_SIZE) -> v2.Compose:
    """Resize + ImageNet normalization; augmentation ONLY when train=True.

    Safe augmentations (plan 1.5): flips, small rotations, brightness/contrast jitter.
    No aggressive crops: they could push a small defect out of the frame and turn a
    defective part into a "good" one.
    """
    steps: list = [v2.ToImage(), v2.Resize(size, antialias=True)]
    if train:
        steps += [
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.RandomRotation(degrees=5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


# --------------------------------------------------------------------------- #
# Dataset and loaders
# --------------------------------------------------------------------------- #
class InspectDataset(Dataset):
    """Returns (image_tensor, label). It does NOT return the mask (avoids leakage)."""

    def __init__(self, df: pd.DataFrame, root: str | Path, transform=None):
        self.root = Path(root)
        self.images = df["image"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = Image.open(self.root / self.images[i]).convert("RGB")  # gray -> 3 channels
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.float32)


def load_mask(root: str | Path, mask_rel: str) -> np.ndarray:
    """Binary mask (0/1). Only for evaluating localization, never as model input."""
    return (np.array(Image.open(Path(root) / mask_rel).convert("L")) > 0).astype(np.uint8)


def pos_weight(df: pd.DataFrame) -> float:
    """n_good / n_defective, for BCEWithLogitsLoss(pos_weight=...)."""
    pos = int(df["label"].sum())
    return (len(df) - pos) / max(pos, 1)


def make_loaders(root: str | Path, splits: dict[str, pd.DataFrame], batch_size: int = 32,
                 size: tuple[int, int] = IMG_SIZE, num_workers: int = 2,
                 balanced_sampler: bool = False, seed: int = SEED) -> dict[str, DataLoader]:
    """train/val/test DataLoaders.

    balanced_sampler=True draws defective parts as often as good ones (WeightedRandomSampler);
    use it INSTEAD of pos_weight in the loss, not together (plan 3.3).
    """
    g = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    loaders = {}
    for name in SPLIT_NAMES:
        df, is_train = splits[name], name == "train"
        ds = InspectDataset(df, root, get_transforms(is_train, size))
        if is_train and balanced_sampler:
            counts = df["label"].value_counts()
            weights = df["label"].map(lambda y: 1.0 / counts[y]).to_numpy()
            sampler = WeightedRandomSampler(weights, num_samples=len(df), replacement=True, generator=g)
            loaders[name] = DataLoader(ds, sampler=sampler, **common)
        else:
            loaders[name] = DataLoader(ds, shuffle=is_train, generator=g if is_train else None, **common)
    return loaders


def make_ae_loaders(root: str | Path, splits: dict[str, pd.DataFrame], batch_size: int = 16,
                    size: tuple[int, int] = IMG_SIZE, num_workers: int = 2,
                    augment: bool = True, seed: int = SEED) -> dict[str, DataLoader]:
    """Loaders for the autoencoder (Phase 4).

    "train_good": ONLY the good parts of train (the autoencoder never sees a defect while training),
                  with augmentation (train only).
    "val":        ALL of validation with labels, no augmentation: used to pick epoch, score variant
                  and threshold. The test split is deliberately NOT built here.
    """
    g = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    good = splits["train"][splits["train"]["label"] == 0].reset_index(drop=True)
    assert len(good) > 0 and good["label"].sum() == 0, "train_good must contain only good parts"
    train_ds = InspectDataset(good, root, get_transforms(augment, size))
    val_ds = InspectDataset(splits["val"], root, get_transforms(False, size))
    return {"train_good": DataLoader(train_ds, shuffle=True, generator=g, **common),
            "val": DataLoader(val_ds, shuffle=False, **common)}
