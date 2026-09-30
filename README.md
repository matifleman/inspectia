# InspectIA

Automatic visual inspection of surface defects in production parts, using computer vision on
[KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/). Final project for the AI course.

**Central question:** on this production line, is it better to use a model trained with defect
examples (supervised CNN) or one that only knows what a good part looks like (autoencoder)?

| Model | Approach |
|---|---|
| Baseline | k-NN / logistic regression on frozen ResNet18 features |
| Main | ResNet18 with transfer learning (supervised) |
| Comparison | Convolutional autoencoder trained only on good parts (anomaly detection) |

Main metric: **F2-score** on the defective class (plus recall, precision, AUC-ROC, average precision).

## Repository layout

```
configs/            one YAML per experiment
scripts/            download_data.py
src/inspectia/      data, models, train, evaluate, utils
notebooks/          one per phase (00_setup, 01_datos, ...)
docs/               practice deliverables, report material, planning docs
```

## Setup

### One-time (per team member)

1. Accept the GitHub collaborator invite and the W&B team invite.
2. Add the shared Drive folder `InspectIA/` to **My Drive** (right click → *Organize* → *Add shortcut*),
   so it is available at `/content/drive/MyDrive/InspectIA` in Colab.
3. In Colab, open **Secrets** (key icon on the left) and add `WANDB_API_KEY` (from
   <https://wandb.ai/authorize>), with notebook access enabled.

### Every session

Open `notebooks/00_setup.ipynb` (or any phase notebook) in Colab with a **GPU runtime** and run the
bootstrap cell. It mounts Drive, clones/pulls the repo, installs dependencies and logs into W&B.

### Local (optional, for editing / linting)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt -e .
python scripts/download_data.py --dest data   # ~850 MB, skipped if already present
```

## Environment

Training runs on Google Colab (T4 GPU). torch/torchvision are taken from Colab, not reinstalled:

| Package | Colab version |
|---|---|
| Python | TODO |
| torch | TODO |
| torchvision | TODO |

## Dataset and license

KolektorSDD2 (ViCoS Lab, University of Ljubljana, with Kolektor Group), licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The dataset is **not**
redistributed in this repo; `scripts/download_data.py` fetches it from the official source.

> Božič, J., Tabernik, D., & Skočaj, D. (2021). Mixed supervision for surface-defect detection:
> from weakly to fully supervised learning. *Computers in Industry*, 129, 103459.
