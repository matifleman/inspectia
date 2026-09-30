"""Download KolektorSDD2 and verify it (idempotent).

Usage:
    python scripts/download_data.py --dest data
    python scripts/download_data.py --dest /content/drive/MyDrive/InspectIA/data

The dataset ends up in <dest>/KolektorSDD2/{train,test}. If it is already there
with the expected counts, nothing is downloaded. Stdlib only, so it can run
before installing requirements.

Dataset: Bozic, Tabernik and Skocaj (2021), "Mixed supervision for
surface-defect detection: from weakly to fully supervised learning".
License: CC BY-NC-SA 4.0 (cite it, do not redistribute).
"""

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"
DATASET_DIR = "KolektorSDD2"
EXPECTED_IMAGES = {"train": 2331, "test": 1004}
MASK_SUFFIX = "_GT"


def check_split(split_dir: Path, expected: int) -> list[str]:
    """Return a list of problems found in one split (empty if it is fine)."""
    if not split_dir.is_dir():
        return [f"{split_dir} does not exist"]
    pngs = list(split_dir.glob("*.png"))
    images = [p for p in pngs if not p.stem.endswith(MASK_SUFFIX)]
    masks = {p.stem for p in pngs if p.stem.endswith(MASK_SUFFIX)}
    problems = []
    if len(images) != expected:
        problems.append(f"{split_dir.name}: {len(images)} images, expected {expected}")
    missing = [p.name for p in images if f"{p.stem}{MASK_SUFFIX}" not in masks]
    if missing:
        problems.append(f"{split_dir.name}: {len(missing)} images without mask, e.g. {missing[:3]}")
    return problems


def verify(root: Path) -> list[str]:
    problems = []
    for split, expected in EXPECTED_IMAGES.items():
        problems += check_split(root / split, expected)
    return problems


def download(url: str, target: Path) -> None:
    last_pct = -1

    def progress(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_pct
        done = min(blocks * block_size, total)
        pct = done * 100 // total if total > 0 else 0
        if pct != last_pct:  # redraw once per percent, not per block
            last_pct = pct
            print(f"\r  {pct:3d}%  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, target, reporthook=progress)
    print()


def remove_exact_copies(root: Path) -> None:
    """Delete '<name> (copy).png' files that are byte-identical to '<name>.png'.

    The official zip ships train/10301 (copy).png and its mask as exact
    duplicates. A copy that differs from its original is kept, so it still
    fails verification and gets looked at.
    """
    for copy in sorted(root.rglob("* (copy).png")):
        original = copy.with_name(copy.name.replace(" (copy)", ""))
        if original.exists() and original.read_bytes() == copy.read_bytes():
            print(f"  removing exact duplicate {copy.relative_to(root)}")
            copy.unlink()


def find_dataset_root(extracted: Path) -> Path:
    """Locate the folder that contains train/ and test/ inside the extracted zip."""
    for candidate in [extracted, *extracted.rglob("*")]:
        if candidate.is_dir() and (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise RuntimeError("train/ and test/ not found inside the zip")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path("data"), help="parent folder for KolektorSDD2/")
    args = parser.parse_args()

    root = args.dest / DATASET_DIR
    if root.exists() and not verify(root):
        print(f"KolektorSDD2 already present and verified at {root}, skipping download.")
        return 0
    if root.exists():
        print(f"{root} exists but is incomplete, re-downloading.")
        shutil.rmtree(root)

    args.dest.mkdir(parents=True, exist_ok=True)
    # Work in a temp folder inside dest so the final move is a cheap rename (also on Drive).
    with tempfile.TemporaryDirectory(dir=args.dest) as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "KolektorSDD2.zip"
        print(f"Downloading {URL}")
        download(URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp / "extracted")
        find_dataset_root(tmp / "extracted").rename(root)
    remove_exact_copies(root)

    problems = verify(root)
    if problems:
        print("Verification FAILED:", *problems, sep="\n  ")
        return 1
    print(f"OK: {EXPECTED_IMAGES['train']} train / {EXPECTED_IMAGES['test']} test images, all with masks, at {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
