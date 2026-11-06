# InspectIA — ESTADO (leer esto PRIMERO, antes que cualquier .py)

Última actualización: 2026-11-06 · Repo oficial: github.com/matifleman/inspectia · Rama de trabajo: `feat/baseline-cnn`.
**Entrega: lunes 16/11/2026, con TODO el plan (Fases 0 a 7), sin recortes (D-023).**

## Documentación
| Archivo | Para qué |
|---|---|
| `docs/ESTADO.md` | Foto actual + mapa del código. Se reescribe al cerrar cada sesión. |
| `docs/DECISIONES.md` | Log append-only (D-001…). Si algo cambia, entrada nueva que reemplaza a la vieja. |
| `docs/RESULTADOS.md` | Solo números reales con su origen. Nada inventado. |
| `docs/GUIA_CONCEPTOS.md`, `docs/GUIA_ESTUDIO.md` | Material de estudio (teoría desde cero + código explicado). |
| `docs/planning/*.docx` | Plan de desarrollo, plan de estudio, stack e idea del proyecto (de Mati). |

## Estado por fase (numeración del plan, D-025)
| Fase | Qué | Estado |
|---|---|---|
| 0 Setup | Repo, entorno, descarga, `00_setup` | Hecha por Mati (scaffold). `00_setup` sin verificar por nosotros. |
| 1 Datos | `01_datos`: auditoría, splits CSV, preprocesamiento | **Hecha** (2026-11-05, Mac local). |
| 2 Protocolo + baseline | `evaluate.py`, `02_baseline` | **Hecha.** Baseline oficial: ResNet18+logística C=0.01, val F2 0.816. |
| 3 CNN | `03_cnn` + configs (ResNet18, sampler, scratch) | **Hecha** (las 3 configs corridas): oficial = `cnn.yaml` (val F2 0.918, AP 0.955). Falta repetir con 3 semillas. |
| 4 Autoencoder | `04_autoencoder` | No empezada. |
| 5 Evaluación final | `05_evaluacion` (test una vez, Grad-CAM, errores) | No empezada. |
| 6 Demo | Gradio + Hugging Face Spaces | No empezada. |
| 7 Cierre | Informe, presentación, README portfolio, repo público | No empezada. |

Resultados reales vigentes: ver `RESULTADOS.md` (todo en validación; ΔF2 CNN−baseline = +0.10, IC [0.035, 0.178]). Test: no se evaluó nunca.

## Mapa del código (`src/inspectia/`)
- `utils.py`: `set_seed`, `get_device` (CUDA > MPS > CPU), `load_config(yaml)`, `init_wandb` (None si no hay wandb).
- `data.py`: `IMG_SIZE=(608,224)` (alto, ancho; imágenes verticales), `build_index(root)` (relativas, `label`, `defect_px`, `width`, `height`; ignora `(copy)`), `make_splits`, `save_splits/load_splits/get_splits` (CSV), `check_splits` (fuga), `get_transforms(train)`, `InspectDataset(df, root)`→(img,label), `load_mask`, `pos_weight`, `make_loaders(..., balanced_sampler)`.
- `audit.py`: `audit_counts`, `check_integrity`, `find_duplicates` (MD5), `brightness_table`, `defect_summary`, gráficos.
- `features.py`: `resnet_features` (512-d, congelada, con caché npz), `hog_features`.
- `models.py`: `DEFAULT_GRIDS`, `make_baseline`, `select_baseline` (por AP en val), `SmallCNN`, `build_resnet18`, `build_model`, `set_stage('head'|'layer4'|'all')`, `freeze_frozen_bn`, `count_trainable`. (Autoencoder: Fase 4.)
- `train.py`: `predict`, `sigmoid`, `fit_stage(... monitor='ap'|'f2')` (restaura mejor época), `save_checkpoint`.
- `evaluate.py`: `evaluate`, `best_f2` / `best_f2_threshold` (**umbral oficial**), `threshold_at_recall` (secundario), `trivial_baselines` (siempre OK / marcar todo), `bootstrap_ci`, `paired_bootstrap_diff` (ΔF2 entre dos modelos), `plot_curves`, `plot_confusion`.
- `configs/*.yaml`: `baseline`, `cnn`, `cnn_sampler`, `cnn_scratch`. `notebooks/`: `00_setup` (Mati), `01_datos`, `02_baseline`, `03_cnn`. `scripts/download_data.py` (Mati; borra duplicados exactos).

## Cómo correr (local en Mac o Colab)
1. Datos en `<repo>/data/KolektorSDD2/{train,test}` (local) o en Drive (`MyDrive/InspectIA/data`, Colab). `scripts/download_data.py --dest data` los baja y verifica.
2. Cada notebook empieza con la celda de arranque común de Mati; después `PATHS`. Local: `pip install pyyaml tqdm wandb scikit-image joblib` en el venv (torch ya instalado).
3. Orden: `01_datos` → `02_baseline` → `03_cnn` (primero `SMOKE=True`).

## Reglas no negociables
Split oficial · test jamás para elegir hiperparámetros/umbral y se usa UNA vez en Fase 5 · máscara no es entrada · augmentation solo en train · nunca accuracy como métrica principal (F2, recall, AUC, PR) · umbral = máx. F2 en validación, fijo (D-024) · un cambio por experimento, registrado en W&B · semilla 42, versiones fijas · dataset fuera del repo (CC BY-NC-SA 4.0).

## Plan de ataque hasta el 16/11 (propuesto, D-023)
| Cuándo | Qué |
|---|---|
| Jue 5 | Migrar al repo y push. Correr `01_datos` y `02_baseline` (rehace el baseline con el tamaño correcto). |
| Vie 6 – Sáb 7 | `03_cnn`: smoke → ResNet18 real → scratch → sampler. Prácticas 6–8. Si el recall queda bajo: mayor resolución / parches / más augmentation. |
| Dom 8 – Lun 9 | Fase 4: `models.Autoencoder`, `04_autoencoder`, score de anomalía, umbral con el mismo protocolo. |
| Mar 10 | Fase 5: `05_evaluacion` (test UNA vez), tabla comparativa, Grad-CAM, análisis de errores, comparación con el paper. |
| Mié 11 | Fase 6: app Gradio + Hugging Face Spaces (delegable al compañero de demo). |
| Jue 12 – Dom 15 | Fase 7: informe, presentación, README de portfolio, repo público. Margen. |
| Lun 16 | Entrega. |
Muy ajustado: el plan original son 10 semanas. Delegar lo marcado "delegable" en el plan (redacción de avances, PCA/figuras, demo, informe) y no perder fines de semana en debugging de entorno. Si hay atraso, el plan recorta en este orden: MVTec AD, anomalib, clustering, demo mínima.

## Abiertos
1. Verificar `00_setup` de Mati en Colab (W&B, Drive compartido) y ver quién entrena en Colab vs Mac.
2. Ejecutar las Fases 1–3 reales (nada se corrió con el tamaño correcto).
3. Pinear la versión de `scikit-image` y `joblib` de Colab en `requirements.txt` (hoy `>=`).
4. Confirmar con el equipo el reparto de tareas delegables.

## Próximo paso
1. Repetir la CNN final (y scratch) con 3 semillas (media ± desvío).
2. Fase 4 (ya se puede empezar): autoencoder. Entorno local: `.venv` en `InspectIA/` con Python 3.14 y torch 2.15 (el plan decía 3.11; sin problemas hasta ahora). VS Code: kernel `InspectIA (.venv)` registrado con ipykernel; abrir la carpeta `InspectIA/`, no `repo/`.
3. Commit/push de `feat/baseline-cnn` y PR a `main` (Mati revisa). `splits/` hay que agregarlo a git.
