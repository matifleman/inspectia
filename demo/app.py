"""Gradio app (Phase 6). Run from the repo root:  python demo/app.py
The same file is copied to the Hugging Face Space by scripts/build_space.py.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE.parent / "src"):          # Space layout / repo layout
    if p.exists():
        sys.path.insert(0, str(p))

import gradio as gr  # noqa: E402

from inspectia.demo import REVIEW_LOW, Inspector  # noqa: E402


def load_inspector() -> Inspector:
    if (HERE / "model" / "cnn_resnet18.pt").exists():           # Hugging Face Space
        return Inspector.from_dir(HERE / "model")
    root = HERE.parent                                          # local repo
    return Inspector.from_artifacts(root / "checkpoints", root / "results")


INSPECTOR = load_inspector()

VERDICT_TEXT = {
    "DEFECTUOSA": "## 🔴 DEFECTUOSA\nSupera el umbral oficial: separar la pieza.",
    "REVISAR": "## 🟠 REVISAR\nZona gris (por debajo del umbral oficial, pero con score alto): que la mire una persona.",
    "OK": "## 🟢 OK\nScore bajo: sin indicios de defecto.",
}


def run(img):
    if img is None:
        return "Subí una imagen de una pieza.", None, None
    r = INSPECTOR.inspect(img)
    text = (f"{VERDICT_TEXT[r.verdict]}\n\n**Probabilidad de defecto:** {r.prob:.3f} · "
            f"**umbral oficial:** {r.threshold:.3f} · **zona de revisión desde:** {REVIEW_LOW:.2f}")
    return text, {"Defectuosa": r.prob, "Buena": 1 - r.prob}, r.overlay


with gr.Blocks(title="InspectIA") as demo:
    gr.Markdown(
        "# InspectIA · inspección visual de defectos\n"
        "ResNet18 con fine-tuning entrenada en KolektorSDD2. El umbral se fijó en validación "
        "maximizando F2 (un falso negativo cuesta mucho más que un falso positivo)."
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Imagen de la pieza")
        with gr.Column():
            verdict = gr.Markdown()
            label = gr.Label(label="Probabilidades")
        heat = gr.Image(label="Grad-CAM (dónde mira la red)")
    gr.Button("Inspeccionar", variant="primary").click(run, inp, [verdict, label, heat])
    gr.Markdown(
        "⚠️ El mapa Grad-CAM se normaliza por imagen: solo es informativo cuando el veredicto es "
        "DEFECTUOSA o REVISAR. Proyecto académico, no usar para decisiones reales de producción.\n\n"
        "Dataset: KolektorSDD2 (ViCoS Lab, CC BY-NC-SA 4.0), Božič, Tabernik & Skočaj (2021), "
        "*Computers in Industry* 129, 103459."
    )

if __name__ == "__main__":
    demo.launch()
