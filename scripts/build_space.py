"""Build a self-contained folder for the Hugging Face Space (Phase 6).

    python scripts/build_space.py --out space

Copies demo/app.py, the `inspectia` package, the official CNN weights and its validation
threshold. It never copies dataset images (CC BY-NC-SA 4.0: cited, not redistributed).
Then: create a Space (SDK Gradio) and upload / `git push` the contents of `space/`.
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="space")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "inspectia", out / "src" / "inspectia",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "demo" / "app.py", out / "app.py")
    shutil.copy(ROOT / "demo" / "requirements.txt", out / "requirements.txt")
    shutil.copy(ROOT / "demo" / "SPACE_README.md", out / "README.md")
    shutil.copy(Path(a.ckpt_dir) / "cnn_resnet18.pt", out / "model" / "cnn_resnet18.pt")
    with open(Path(a.results_dir) / "cnn_resnet18_val.json") as f:
        j = json.load(f)
    with open(out / "model" / "model_info.json", "w") as f:
        json.dump({"threshold": j["threshold"], "val": j["val"], "source": "cnn_resnet18 (seed 42), val max-F2"}, f, indent=2)
    print(f"Space ready in {out}/ (threshold {j['threshold']:.4f})")


if __name__ == "__main__":
    main()
