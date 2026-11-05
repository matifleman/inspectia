"""Dataset audit (Phase 1.1 and 1.2, Practice 3): counts, integrity, duplicates, brightness,
defect size and example grids. Everything works on the DataFrame from `data.build_index`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def audit_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Images per split and class (0 = good, 1 = defective) plus the defect rate."""
    table = pd.crosstab(df["split"], df["label"], margins=True)
    print(table)
    print("\n% defective:\n", df.groupby("split")["label"].mean().mul(100).round(1).to_string())
    print("\nImage sizes (width, height):", df.groupby(["width", "height"]).size().shape[0],
          "distinct | width", df["width"].min(), "-", df["width"].max(),
          "| height", df["height"].min(), "-", df["height"].max())
    return table


def check_integrity(df: pd.DataFrame, root: str | Path) -> dict[str, list[str]]:
    """Unreadable files and masks whose size differs from the image's."""
    root = Path(root)
    corrupt, mismatch = [], []
    for row in df.itertuples():
        try:
            with Image.open(root / row.image) as im:
                im.verify()
            with Image.open(root / row.mask) as mk:
                mk.verify()
        except Exception:  # noqa: BLE001 - any failure means the file is unusable
            corrupt.append(row.image)
            continue
        with Image.open(root / row.mask) as mk:
            if mk.size != (row.width, row.height):
                mismatch.append(row.image)
    print(f"Corrupt files: {len(corrupt)} | masks with a different size than the image: {len(mismatch)}")
    return {"corrupt": corrupt, "mask_size_mismatch": mismatch}


def find_duplicates(df: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Byte-identical images (MD5), within a split or ACROSS splits (leakage risk)."""
    root = Path(root)
    hashes = {row.image: hashlib.md5((root / row.image).read_bytes()).hexdigest()
              for row in df.itertuples()}
    groups = pd.Series(hashes).rename("md5").reset_index().rename(columns={"index": "image"})
    groups["split"] = groups["image"].str.split("/").str[0]
    dup = groups[groups.duplicated("md5", keep=False)].sort_values("md5")
    cross = dup.groupby("md5")["split"].nunique().gt(1).sum() if len(dup) else 0
    print(f"Duplicate images: {len(dup)} in {dup['md5'].nunique() if len(dup) else 0} group(s); "
          f"groups spanning train and test: {cross}")
    return dup.reset_index(drop=True)


def brightness_table(df: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Mean and std of the gray level of every image (to spot lighting outliers)."""
    root = Path(root)
    rows = []
    for row in df.itertuples():
        a = np.asarray(Image.open(root / row.image).convert("L"), dtype=np.float32)
        rows.append(dict(image=row.image, split=row.split, label=row.label,
                         mean=float(a.mean()), std=float(a.std())))
    return pd.DataFrame(rows)


def defect_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Defect area in pixels and as % of the image: it decides the input resolution."""
    d = df[df["label"] == 1].copy()
    d["defect_pct"] = 100 * d["defect_px"] / (d["width"] * d["height"])
    print(d[["defect_px", "defect_pct"]].describe(percentiles=[.05, .25, .5, .75, .95]).round(3))
    return d


def plot_examples(df: pd.DataFrame, root: str | Path, n: int = 4, seed: int = 42):
    """Grid: n good parts on top, n defective ones below with the mask outlined in red."""
    import matplotlib.pyplot as plt

    root = Path(root)
    rng = np.random.RandomState(seed)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 11))
    for r, label in enumerate((0, 1)):
        sub = df[(df["label"] == label) & (df["split"] == "train")]
        for ax, (_, row) in zip(axes[r], sub.iloc[rng.permutation(len(sub))[:n]].iterrows()):
            img = np.asarray(Image.open(root / row["image"]).convert("L"))
            ax.imshow(img, cmap="gray")
            if label == 1:
                mask = np.asarray(Image.open(root / row["mask"]).convert("L")) > 0
                ax.contour(mask, levels=[0.5], colors="red", linewidths=1)
            ax.set_title(f"{Path(row['image']).stem}\n{'DEFECT' if label else 'OK'}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    return fig


def plot_defect_sizes(defects: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    a.hist(defects["defect_px"], bins=40)
    a.set(title="Defect area (pixels)", xlabel="pixels", ylabel="images")
    b.hist(defects["defect_pct"], bins=40)
    b.set(title="Defect area (% of image)", xlabel="%")
    fig.tight_layout()
    return fig


def plot_brightness(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, name in ((0, "good"), (1, "defective")):
        ax.hist(table.loc[table["label"] == label, "mean"], bins=40, alpha=0.6, label=name, density=True)
    ax.set(title="Mean gray level per image", xlabel="mean brightness (0-255)")
    ax.legend()
    fig.tight_layout()
    return fig
