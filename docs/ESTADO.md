# InspectIA — ESTADO (leer esto PRIMERO, antes que cualquier .py)

> Si sos un agente de IA: leé también `CLAUDE.md` (reglas) en la raíz del repo.

Última actualización: 2026-11-06 (cierre de Fase 5; handoff al equipo) · Repo oficial: github.com/matifleman/inspectia · Rama de trabajo: `feat/baseline-cnn`.
**Entrega: lunes 16/11/2026, con TODO el plan (Fases 0 a 7), sin recortes (D-023).**

## Documentación
| Archivo | Para qué |
|---|---|
| `docs/ESTADO.md` | Foto actual + mapa del código. Se reescribe al cerrar cada sesión. |
| `docs/DECISIONES.md` | Log append-only (D-001…). Si algo cambia, entrada nueva que reemplaza a la vieja. |
| `docs/RESULTADOS.md` | Solo números reales con su origen. Nada inventado. |
| `docs/GUIA_CONCEPTOS.md`, `docs/GUIA_ESTUDIO.md` | Material de estudio (teoría desde cero + código explicado). |
| `CLAUDE.md` / `AGENTS.md` | Reglas del proyecto para agentes de IA. |
| `docs/planning/*.docx` | Plan de desarrollo, plan de estudio, stack e idea del proyecto (de Mati). |

## Estado por fase (numeración del plan, D-025)
| Fase | Qué | Estado |
|---|---|---|
| 0 Setup | Repo, entorno, descarga, `00_setup` | Hecha por Mati (scaffold). `00_setup` sin verificar por nosotros (tiene un cambio local sin commitear; Mati decide). |
| 1 Datos | `01_datos`: auditoría, splits CSV, preprocesamiento | **Hecha** (2026-11-05, Mac local). |
| 2 Protocolo + baseline | `evaluate.py`, `02_baseline` | **Hecha.** Baseline oficial: ResNet18+logística C=0.01, val F2 0.816. |
| 3 CNN | `03_cnn` + configs (ResNet18, sampler, scratch) | **Hecha** (las 3 configs corridas): oficial = `cnn.yaml` (val F2 0.918, AP 0.955). 3 semillas hechas: F2 0.923 ± 0.005, AP 0.956 ± 0.001. |
| 4 Autoencoder | `04_autoencoder` + `configs/autoencoder.yaml` | **Hecha:** oficial `autoencoder_b2.yaml`, 3 semillas: val F2 0.733 ± 0.011, AP 0.792 ± 0.005 (siempre época 1). |
| 5 Evaluación final | `05_evaluacion` (test una vez, Grad-CAM, errores) | **Hecha** (test evaluado una vez el 2026-11-06). Cifras oficiales en RESULTADOS.md (CNN F2 0.834 ± 0.013). Pendiente: revisión manual de FP/FN grandes y comparación con el paper. |
| 6 Demo | Gradio local + Space Static en Hugging Face | **Hecha.** Space público (Static + ONNX Runtime Web, D-049): https://huggingface.co/spaces/ipejinakis/inspetIA (estado Running el 2026-11-06). App Gradio local: `python demo/app.py` (D-048). |
| 7 Cierre | Informe, presentación, README portfolio, repo público | **No empezada** (ver "Qué falta"). |

Resultados reales vigentes: ver `RESULTADOS.md` (todo en validación; ΔF2 CNN−baseline = +0.10, IC [0.035, 0.178]). Test: evaluado UNA vez (no se vuelve a usar para decidir nada).

## Mapa del código (`src/inspectia/`)
- `utils.py`: `set_seed`, `get_device` (CUDA > MPS > CPU), `load_config(yaml)`, `init_wandb` (None si no hay wandb).
- `data.py`: `IMG_SIZE=(608,224)` (alto, ancho; imágenes verticales), `build_index`, `make_splits`, `save_splits/load_splits/get_splits` (CSV), `check_splits` (fuga), `get_transforms(train)`, `InspectDataset`, `load_mask`, `pos_weight`, `make_loaders(..., balanced_sampler)`, `make_ae_loaders` (train solo buenas + val; no construye test).
- `audit.py`: auditoría de datos (conteos, integridad, duplicados MD5, brillo, defectos).
- `features.py`: `resnet_features` (512-d, congelada, caché npz), `hog_features`.
- `models.py`: `make_baseline`, `select_baseline` (por AP en val), `SmallCNN`, `ConvAutoencoder`/`build_autoencoder`, `build_resnet18`, `build_model`, `set_stage('head'|'layer4'|'all')`, `freeze_frozen_bn`.
- `train.py`: `predict`, `sigmoid`, `fit_stage`, `save_checkpoint`; autoencoder: `error_map`, `score_variants` (mean/top5/top1/max), `anomaly_scores` (topk en CPU: en MPS es inestable), `fit_autoencoder`.
- `evaluate.py`: `evaluate`, `best_f2_threshold` (**umbral oficial**), `threshold_at_recall`, `trivial_baselines`, `bootstrap_ci`, `paired_bootstrap_diff`, `plot_curves`, `plot_confusion`.
- `demo.py` (Fase 6): `Inspector` (CNN + umbral de val + Grad-CAM → `Inspection`), `decide` (DEFECTUOSA/REVISAR/OK), `overlay_cam`. UI en `demo/app.py`; `scripts/build_space.py` arma `space/` (solo útil con HF PRO). Space gratis: `scripts/export_onnx.py` + `web/index.html` → `space_static/` (D-049).
- `final.py` (Fase 5): `build_registry`, `score_all`, `f2_curve`. `explain.py`: `gradcam(model,x,layer)`, `cam_vs_mask`.
- `configs/`: `baseline`, `cnn` (oficial), `cnn_sampler`, `cnn_scratch`, `autoencoder`, `autoencoder_b4`, `autoencoder_b2` (oficial AE).
- `notebooks/`: `00_setup` (Mati), `01_datos`, `02_baseline`, `03_cnn` (variable `CONFIG` y `SEED_OVERRIDE`), `04_autoencoder`, `05_evaluacion`, `06_demo`. `scripts/download_data.py`, `scripts/build_space.py`.

## Cómo correr (Colab T4 oficial, o Mac local)
1. Datos en `<repo>/data/KolektorSDD2/{train,test}` o en Drive. `python scripts/download_data.py --dest data` los baja.
2. Cada notebook empieza con la celda de arranque común y después `PATHS`. Correr "Run All" desde la primera celda (si no, `ModuleNotFoundError: inspectia`).
3. Orden: `01_datos` → `02_baseline` → `03_cnn` → `04_autoencoder` → `05_evaluacion`. Primero `SMOKE=True`.
4. **No hace falta reentrenar para la demo ni el informe**: usar los artefactos de abajo.
5. NO volver a correr `05_evaluacion` desde cero: el test ya se evaluó (lock en `results/test_scores.npz`).
6. Demo local (Fase 6): `pip install gradio` en el venv y `python demo/app.py`. Notebook `06_demo` prueba la lógica sin interfaz.
7. Demo pública (Space Static, gratis): `python scripts/export_onnx.py --out space_static` (necesita `onnx` y `onnxruntime`; verifica equivalencia con PyTorch), probar con `cd space_static && python -m http.server 8000` y subir los 4 archivos al Space. `space/` y `space_static/` están en `.gitignore`. Gradio Spaces en Hugging Face requieren plan pago (D-049).

## Artefactos que no están en git (gitignored)
Subidos el 2026-11-06 a la carpeta compartida de Drive `InspectIA/artefactos/{checkpoints,results}/` (misma estructura que el repo). Para usarlos: copiar `artefactos/checkpoints` → `repo/checkpoints/` y `artefactos/results` → `repo/results/` (en Colab con Drive montado: `cp -r`).
| Archivo | Para qué | Tamaño |
|---|---|---|
| `checkpoints/cnn_resnet18.pt` (+ `_s1`, `_s2`) | CNN oficial, demo (Fase 6) y Grad-CAM | 45 MB c/u |
| `checkpoints/baseline.joblib` | Baseline oficial | 16 KB |
| `checkpoints/autoencoder_b2.pt` (+ `_s1`, `_s2`) | AE oficial | 0.8 MB c/u |
| `results/*_val.json` | Umbral y métricas de validación por modelo (lo lee `build_registry`) | KB |
| `results/test_results.json`, `test_scores.npz` | Resultado de test (único) | KB |
Para la demo, el umbral de decisión de la CNN sale de `results/cnn_resnet18_val.json` (máx. F2 en val). Los números oficiales ya están copiados en `RESULTADOS.md`.

## Reglas no negociables
Ver `CLAUDE.md`. Resumen: split oficial · test jamás para decidir (ya usado una vez) · máscara no es entrada · augmentation solo en train · nunca accuracy (F2, recall, AUC, PR) · umbral = máx. F2 en val, fijo · semilla 42, versiones fijas · dataset fuera del repo.

## Resultados oficiales (detalle y origen en RESULTADOS.md)
Test (1004 imágenes, 110 defectuosas, umbral fijado en val): CNN F2 0.834 ± 0.013 (AP 0.916) · baseline F2 0.685 (AP 0.780) · autoencoder F2 0.685 ± 0.051 (AP 0.744). Limitaciones a declarar en el informe: brecha val→test de la CNN, umbral de val inestable entre semillas (0.30–0.66), AE con mejor época siempre 1, val con solo 49 defectuosas.

## Qué falta (hasta el lunes 16/11) y quién lo hace
| # | Tarea | Detalle | Responsable |
|---|---|---|---|
| 1 | Comparación con el paper | Copiar de Božič, Tabernik y Skočaj (2021) la tabla de KolektorSDD2 y el protocolo (métrica, split, resolución, umbral) y completar sección 9 de `05_evaluacion.ipynb` + `RESULTADOS.md`. No inventar números. PDF: prints.vicos.si/publications/385 · arXiv 2104.06064. Aclarar diferencias de protocolo (F2/AP con umbral de val vs lo que reportan). | _a asignar_ |
| 2 | Revisión manual de errores | Mirar a ojo FP de mayor score (20245, 20651, 20002, 20846) y FN grandes (20141, 20625, 20969, 20772): ¿ruido de etiqueta, defecto invisible o fallo real? Anotar en RESULTADOS.md. | _a asignar_ |
| 3 | Fase 6: demo | **Hecha.** URL: https://huggingface.co/spaces/ipejinakis/inspetIA. Pendiente solo: probar el Space con una buena y una defectuosa (confirmado en localhost). | Juan |
| 4 | Fase 7: informe | Estructura y tono del plan. Insumos: RESULTADOS.md, DECISIONES.md, figuras del `05_evaluacion`, `GUIA_CONCEPTOS.md`. Declarar limitaciones y que el test se usó una vez. | _a asignar_ |
| 5 | Fase 7: presentación y README portfolio | README en inglés orientado a portfolio, repo público (dataset citado, no redistribuido). | _a asignar_ |
| 6 | Prácticas 2 a 8 | Redacción de la consigna según `docs/planning` y el material ya escrito. | _a asignar_ |
| 7 | Verificar `00_setup` en Colab | W&B, Drive compartido, quién entrena en Colab. Decidir qué hacer con el cambio local de `00_setup.ipynb`. | Mati |
| 8 | Housekeeping | Pinear `scikit-image` y `joblib` (hoy `>=`) con las versiones de Colab. Abrir PR `feat/baseline-cnn` → `main` y aprobar. | Mati / Juan |

Si hay atraso, el plan recorta en este orden: MVTec AD, anomalib, clustering, demo mínima (la demo mínima = score + decisión, sin Grad-CAM).

## Cómo retomar con un agente de IA
1. `git clone https://github.com/matifleman/inspectia && cd inspectia && git checkout feat/baseline-cnn` (o `main` una vez mergeado el PR).
2. Bajar los artefactos de Drive a `checkpoints/` y `results/`; bajar los datos con `scripts/download_data.py`.
3. Abrir el agente en la raíz (lee `CLAUDE.md`), pedirle que lea `docs/ESTADO.md` y asignarle una fila de "Qué falta".
4. Al terminar: actualizar ESTADO.md/RESULTADOS.md/DECISIONES.md y hacer PR.
