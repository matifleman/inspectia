# InspectIA — Resultados reales (solo números con corrida detrás)

Todo en VALIDACIÓN (467 imágenes, 49 defectuosas = 20 % estratificado del train, seed 42, entrada 608×224 correcta).
**TEST: evaluado UNA vez el 2026-11-06 (`05_evaluacion`, D-026/D-045). Ver la sección "Fase 5 — TEST" al final; esas son las cifras a citar en el informe.**
`results/` y `checkpoints/` no van a git (están en `.gitignore`): este archivo es el registro de los números.
Los resultados anteriores al fix de orientación (D-020) se eliminaron de acá; siguen en el historial de git.

## Cómo leer estos números (sesgo conocido)
La época, el hiperparámetro y el umbral se eligieron en la MISMA validación donde se reporta, y con 49 defectuosas un error mueve el recall ~2 puntos. Todo es **optimista**; los IC bootstrap muestran el ruido pero no corrigen el sesgo de selección. La vara honesta es el test de la Fase 5.

## Fase 1 — Datos (2026-11-05, `01_datos`)
Train 2331 (2085 buenas / 246 def.) · test 1004 (894 / 110) · 0 corruptas · 0 duplicados · split propio: train 1864 (197 def.) / val 467 (49) / test 1004 (110).
Imágenes verticales: ancho 184–241, alto 597–665 (601 tamaños distintos). Brillo medio ≈ 44/255, igual en ambas clases.
Defectos: mediana 2273 px (1,56 % de la imagen), percentil 5 = 232 px, mínimo 23 px. Los más chicos son el riesgo para un ResNet que reduce 32×.

## Fase 2 — Baselines, validación (2026-11-05, `02_baseline`)
| Modelo | Hiperparámetro (por AP en val) | Umbral | F2 [IC 95 %] | Recall | Precision | AUC | AP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet18 + logística** (oficial) | C=0.01 | 0.671 | **0.816** [0.71–0.90] | 0.816 [0.69–0.92] | 0.816 | 0.962 | 0.880 | 9 | 9 |
| ResNet18 + kNN | k=41 | 0.049 | 0.594 [0.49–0.69] | 0.694 | 0.378 | 0.803 | 0.623 | 15 | 56 |
| HOG + logística | C=1 | ~0 | 0.370 (degenerado) | 1.0 | 0.105 | 0.510 | 0.168 | 0 | 417 |
| HOG + kNN | k=41 | 0 | 0.370 (degenerado) | 1.0 | 0.105 | 0.523 | 0.114 | 0 | 418 |

HOG ≈ azar (AUC ≈ 0.5): descartado, sin más tuning. Referencias triviales en val: "siempre OK" accuracy 0.895 / F2 0 · "marcar todo" precision 0.105 / **F2 0.370**.

## Fase 3 — CNN, validación (2026-11-06, `03_cnn`, `cnn.yaml`, MPS, seed 42, 1 corrida)
ResNet18 preentrenada, `pos_weight` ≈ 8.46, etapas head (lr 1e-3, 8 ép.) → layer4 (lr 1e-4, early stopping paciencia 3 por AP; cortó en la época 5, mejor = época 2).
| Modelo | Umbral | F2 [IC 95 %] | Recall | Precision | AUC | AP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| **CNN ResNet18 (fine-tuning)** | 0.663 | **0.918** [0.85–0.98] | 0.918 [0.84–0.98] | 0.918 | 0.981 | 0.955 | 4 | 4 |

- Etapa head sola: AP 0.748 / F2 0.709 / AUC 0.906 (todavía mejoraba; subentrenada). Al descongelar layer4: AP 0.949 / F2 0.909 desde la primera época.
- Pico de `train_loss` 4.71 en la 1.ª época de layer4 (efecto de descongelar); se recupera solo (≈1.0 desde la 2.ª).
- Sobreajuste leve: val_loss mínimo en la época 2 (0.251) y sube a 0.31; el early stopping restauró la mejor.
- Tiempo: ≈13 s/época (head) y ≈18 s/época (layer4) en Mac MPS; corrida completa ≈ 4 min.
- **CNN vs baseline** (bootstrap pareado estratificado, 2000 remuestreos, cada modelo con su umbral fijo): ΔF2 = **+0.102**, IC 95 % **[+0.035, +0.178]**, P(ΔF2 ≤ 0) = 0.002. El IC excluye 0 (en val, con el sesgo de selección de arriba; una sola semilla).

### Experimentos de control de la Fase 3 (2026-11-06, mismas condiciones, 1 corrida, seed 42)
| Config | Cambio vs `cnn.yaml` | Umbral | F2 [IC 95 %] | Recall | Precision | AUC | AP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| `cnn.yaml` (oficial) | — | 0.663 | **0.918** [0.85–0.98] | 0.918 | 0.918 | 0.981 | 0.955 | 4 | 4 |
| `cnn_sampler.yaml` | `WeightedRandomSampler` en vez de `pos_weight` | 0.892 | 0.918 [0.85–0.98] | 0.918 | 0.918 | 0.989 | 0.961 | 4 | 4 |
| `cnn_scratch.yaml` | CNN chica (3 bloques conv) desde cero, 20 ép. | 0.432 | 0.849 [0.77–0.91] | 0.898 | 0.698 | 0.946 | 0.873 | 5 | 19 |

- **Sampler vs pos_weight:** ΔF2 = 0.000, IC [−0.050, +0.050]. Empate; 3 de los 4 FN (val idx 331, 401, 464) y 2 de los 4 FP son las mismas imágenes en ambos. Se mantiene `pos_weight` (más simple). Resultado negativo para el informe. Los umbrales no son comparables entre configuraciones (el sampler desplaza las probabilidades).
- **Pretrained vs scratch:** ΔF2 (pretrained − scratch) = +0.069, IC [−0.022, +0.164], P(ΔF2 ≤ 0) = 0.06: **no concluyente en F2**. Las métricas independientes del umbral sí favorecen a la preentrenada (AUC 0.981 vs 0.946, AP 0.955 vs 0.873) y su entrenamiento es mucho más estable.
- La CNN chica desde cero sorprende: F2 0.849 supera al baseline (0.816) con solo 197 defectuosas. Sugiere que los defectos son manchas de contraste alto y fáciles de aprender. Su entrenamiento es inestable (val_loss con pico a 3.39 en la ép. 9, F2 oscilando entre 0.76 y 0.85); la mejor época (10) es en buena parte suerte, así que su F2 está más inflado por la selección que el de la ResNet18.
- La hipótesis de la Práctica 7 ("el fine-tuning supera a la CNN desde cero") se sostiene en AP/AUC y en estabilidad, pero no se puede afirmar en F2 con esta validación.

### CNN oficial (`cnn.yaml`) con 3 semillas (2026-11-06; el split es siempre el mismo, cambia init y orden de lotes)
| Semilla | Umbral | F2 | Recall | Precision | AUC | AP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.663 | 0.918 | 0.918 | 0.918 | 0.981 | 0.955 | 4 | 4 |
| 1 | 0.600 | 0.927 | 0.939 | 0.885 | 0.989 | 0.957 | 3 | 6 |
| 2 | 0.304 | 0.924 | 0.939 | 0.868 | 0.978 | 0.957 | 3 | 7 |
| **media ± desvío** | 0.52 ± 0.19 | **0.923 ± 0.005** | 0.932 ± 0.012 | 0.890 ± 0.026 | 0.983 ± 0.006 | 0.956 ± 0.001 | 3.3 ± 0.6 | 5.7 ± 1.5 |

- La variación entre semillas (≈ 0.005 en F2, 0.001 en AP) es mucho menor que la distancia al baseline (F2 0.816, AP 0.880): el resultado no depende de la semilla.
- **El umbral sí es inestable** (0.30–0.66): el F2 en val es una meseta muy plana con 49 defectuosas, y el máximo cae en puntos distintos según la corrida. Para la Fase 5: fijar el umbral de cada modelo en val (D-024) pero considerar usar el de una corrida representativa o reportar la sensibilidad al umbral (curva F2 vs umbral).

## Fase 4 — Autoencoder, validación (2026-11-06, `04_autoencoder`, `autoencoder.yaml`, MPS, seed 42, 1 corrida)
ConvAutoencoder (202 mil parámetros, cuello 38×14×16), entrenado solo con las 1667 buenas del train (MSE). Early stopping por AP de val (paciencia 6): cortó en la época 7, **mejor = época 1**.
| Variante de score | AP val | AUC val |
|---|---|---|
| mean | 0.143 | 0.556 |
| top5 | 0.667 | 0.834 |
| **top1** (elegida, mejor AP) | **0.782** | **0.882** |
| max | 0.749 | 0.875 |

| Modelo | Umbral | F2 [IC 95 %] | Recall | Precision | AUC | AP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Autoencoder (top1) | 0.231 | 0.723 [0.61–0.83] | 0.694 [0.57–0.82] | 0.872 [0.78–0.97] | 0.882 | 0.782 | 15 | 5 |

- Sin ninguna etiqueta de defecto en el entrenamiento, supera claramente a "marcar todo" (F2 0.37) y al HOG, pero queda por debajo del baseline supervisado (F2 0.816, AP 0.880) y de la CNN (F2 0.923, AP 0.956).
- **El promedio del error (`mean`) casi no sirve** (AP 0.14): el defecto es chico y se diluye. Los scores que miran los píxeles peor reconstruidos sí (top1 > max > top5). Confirma la hipótesis de diseño.
- **Hallazgo clave:** el AP de val BAJA a medida que el autoencoder aprende mejor (época 1: 0.782 → época 7: 0.747, mientras val_mse baja de 0.041 a 0.017). Aprende a reconstruir también los defectos (generaliza demasiado). El early stopping eligió la época 1, es decir una red casi sin entrenar: resultado frágil y poco defendible tal cual. Siguiente experimento: cuello más angosto (`autoencoder_b4.yaml`, `autoencoder_b2.yaml`).

### Autoencoder: barrido del cuello 16 / 4 / 2 (2026-11-06, seed 42, 1 corrida cada una)
| Config | Mejor época | Variante | Umbral | F2 [IC 95 %] | Recall | Precision | AUC | AP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|---|
| `autoencoder.yaml` (cuello 16) | 1 | top1 | 0.231 | 0.723 [0.61–0.83] | 0.694 | 0.872 | 0.882 | 0.782 | 15 | 5 |
| `autoencoder_b4.yaml` (cuello 4) | 1 | top1 | 0.149 | 0.717 [0.61–0.81] | 0.714 | 0.729 | 0.885 | 0.789 | 14 | 13 |
| `autoencoder_b2.yaml` (cuello 2) | 1 | top1 | 0.123 | 0.744 | 0.796 | 0.591 | 0.901 | 0.798 | 10 | 27 |

- **Hipótesis refutada:** achicar el cuello (16 → 4 → 2) apenas mueve el AP (0.782 → 0.789 → 0.798, dentro del ruido; el cuello 2 gana recall 0.80 a costa de 27 FP, precision 0.59). El `val_mse` final es casi igual (0.018 vs 0.017) y el patrón se repite: el mejor AP es en la época 1 y baja al entrenar más (0.789 → 0.739). El ancho del cuello no es el factor limitante.
- Lectura: el AP de ~0.78 viene en buena parte de que los defectos tienen más contraste que la textura (una red casi sin entrenar ya marca "lo que se aparta de un fondo suave"); entrenar más no la hace mejor detectora de anomalías sino mejor copiadora. El ruido granular de la textura domina el MSE.

### Autoencoder oficial (`autoencoder_b2.yaml`, cuello 2) con 3 semillas (2026-11-06; split fijo)
| Semilla | Umbral | F2 | Recall | Precision | AUC | AP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.123 | 0.744 | 0.796 | 0.591 | 0.901 | 0.798 | 10 | 27 |
| 1 | 0.151 | 0.723 | 0.714 | 0.761 | 0.887 | 0.787 | 14 | 11 |
| 2 | 0.125 | 0.731 | 0.776 | 0.594 | 0.898 | 0.791 | 11 | 26 |
| **media ± desvío** | 0.133 ± 0.015 | **0.733 ± 0.011** | 0.762 ± 0.042 | 0.649 ± 0.097 | 0.895 ± 0.007 | **0.792 ± 0.005** | 11.7 ± 2.1 | 21.3 ± 9.0 |

- AP y F2 son estables entre semillas; lo que se mueve es el equilibrio recall/precisión (el umbral cae en puntos distintos de la meseta). Las 3 semillas eligen la época 1 y la variante `top1`.
- Comparación en validación (media de 3 semillas): CNN F2 0.923 / AP 0.956 · baseline F2 0.816 / AP 0.880 · **autoencoder F2 0.733 / AP 0.792**. El orden CNN > baseline > autoencoder se sostiene, con la salvedad del sesgo de selección en val.

## Fase 5 — TEST oficial (2026-11-06, `05_evaluacion`, 1004 imágenes, 110 defectuosas)
Protocolo D-045: cada modelo con SU umbral fijado en validación; nada se reajustó con el test. Referencias triviales en test: "siempre OK" F2 0 · "marcar todo" precision 0.110, **F2 0.381**.

| Modelo | Umbral (val) | F2 | Recall | Precision | AUC | AP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Baseline (ResNet18 + logística) | 0.671 | 0.685 [0.60–0.77] | 0.664 | 0.785 | 0.899 | 0.780 | 37 | 20 |
| CNN, semilla 42 | 0.663 | 0.822 [0.76–0.89] | 0.800 | 0.926 | 0.966 | 0.914 | 22 | 7 |
| CNN, semilla 1 | 0.600 | 0.830 | 0.818 | 0.882 | 0.969 | 0.918 | 20 | 12 |
| CNN, semilla 2 | 0.304 | 0.849 | 0.845 | 0.861 | 0.963 | 0.917 | 17 | 15 |
| **CNN, media ± desvío** | | **0.834 ± 0.013** | 0.821 ± 0.023 | 0.890 ± 0.033 | 0.966 ± 0.003 | **0.916 ± 0.002** | 19.7 ± 2.5 | 11.3 ± 4.0 |
| Autoencoder b2, semilla 42 | 0.123 | 0.715 [0.65–0.78] | 0.773 | 0.552 | 0.911 | 0.747 | 25 | 69 |
| Autoencoder b2, semilla 1 | 0.151 | 0.626 | 0.618 | 0.660 | 0.900 | 0.728 | 42 | 35 |
| Autoencoder b2, semilla 2 | 0.125 | 0.713 | 0.745 | 0.607 | 0.917 | 0.758 | 28 | 53 |
| **Autoencoder, media ± desvío** | | **0.685 ± 0.051** | 0.712 ± 0.082 | 0.607 ± 0.054 | 0.909 ± 0.008 | **0.744 ± 0.015** | 31.7 ± 9.1 | 52.3 ± 17.0 |

**De validación a test (sesgo de selección):** F2 cae 0.132 (baseline), 0.075–0.097 (CNN, las 3 semillas) y 0.018–0.097 (autoencoder). AP cae de 0.880 → 0.780 (baseline), 0.956 → 0.916 (CNN), 0.792 → 0.744 (autoencoder). El orden se mantiene entre CNN y el resto, pero el baseline cae más y empata con el autoencoder.

**Diferencias (bootstrap pareado de ΔF2 en test, semilla 42):**
| Comparación | ΔF2 | IC 95 % | Veredicto |
|---|---|---|---|
| CNN − baseline | +0.138 | [+0.082, +0.200] | real |
| Autoencoder − baseline | +0.031 | [−0.056, +0.117] | no concluyente (empate) |
| CNN − autoencoder | +0.107 | [+0.041, +0.179] | real |

**Recall por tamaño de defecto (test; detectadas/total):**
| Tamaño | Baseline | CNN (s42) | Autoencoder (s42) |
|---|---|---|---|
| < 250 px | 0/5 | 0/5 | 3/5 |
| 250–1000 px | 4/19 | 11/19 | 12/19 |
| 1000–3000 px | 26/36 | 31/36 | 28/36 |
| > 3000 px | 43/50 | 46/50 | 42/50 |

**Errores de la CNN (s42):** 22 FN, 7 FP. 13 de los 22 FN son defectos < 1000 px (el 22 % de los defectos explica el 59 % de los FN); 9 FN son ≥ 1000 px, incluidos 4 de más de 3000 px con score bajo (imágenes 20141, 20625, 20969, 20772), que no se explican por el tamaño. Los 4 FP de mayor score (20245, 20651, 20002, 20846; scores 0.91–0.98) tienen un punto brillante en el centro del mapa de calor: hay que revisarlos a mano, pueden ser defectos mínimos sin anotar (ruido de etiquetas).

**Grad-CAM (CNN s42, 88 defectuosas detectadas):** el pico del mapa cae dentro del defecto en el 69 %; el 26 % de la energía cae en la máscara, que ocupa en promedio el 3,6 % de la imagen (≈ 7× el azar). En varios FN pequeños (20228, 20231, 20275, 20259) el mapa sí se enciende sobre el defecto pero el score queda por debajo del umbral: la red lo ve, no llega a superar el umbral fijado en validación.

**Sensibilidad al umbral (solo informativa, NO se usó para elegir nada):** en test la curva de F2 de la CNN tiene una meseta ancha (≈ 0.80–0.86 entre umbrales 0.2 y 0.9); el baseline es más puntiagudo y el autoencoder muy puntiagudo (su umbral fijado en val queda a la izquierda de su pico en test).

## Fase 6 — Demo (2026-11-06, `scripts/export_onnx.py`, `web/index.html`, `demo/app.py`)
Qué es y por qué: ver D-048 (Gradio local, tres veredictos) y D-049 (Space Static con ONNX Runtime Web porque Gradio es pago en Hugging Face). Space público: https://huggingface.co/spaces/ipejinakis/inspetIA (estado Running el 2026-11-06). Usa la CNN oficial `cnn_resnet18.pt` (semilla 42), umbral de validación 0.6633 (máx. F2), zona REVISAR desde 0.30 (criterio de producto, no validado con test).

**Verificación de equivalencia** (salida de `export_onnx.py`, 6 imágenes de VALIDACIÓN: 3 defectuosas + 3 buenas; diferencia máxima absoluta):
| Comparación | Diferencia máx. |
|---|---|
| logit del wrapper vs modelo original | 0.0 |
| Grad-CAM = ReLU(Σ w_c·A_c) vs `explain.gradcam` (normalizado) | 2.98e-07 |
| logit ONNX vs PyTorch | 7.63e-05 |
| Grad-CAM ONNX vs PyTorch | 3.52e-06 |
El Grad-CAM calculado así es exacto porque tras `layer4` hay average-pool + Linear(512→1).

**Prueba manual en el navegador (localhost, 2026-11-06)** con dos imágenes de TRAIN (la red las vio al entrenar: solo prueba que la app anda, no mide rendimiento):
| Imagen | Real | Veredicto | Probabilidad | Grad-CAM |
|---|---|---|---|---|
| `train/10628.png` | defectuosa (3863 px) | DEFECTUOSA | 1.000 | cae sobre el defecto (borde izquierdo) |
| `train/11318.png` | buena | OK | 0.047 | dibuja zonas igual (se normaliza por imagen; solo informativo si el veredicto no es OK) |
**Consistencia de `06_demo.ipynb` (2026-11-13, CPU, notebook ejecutado y guardado con salidas):** el pipeline de la demo (imagen suelta → probabilidad) reproduce los scores de validación guardados en la Fase 3: `max |diff|` = 4.17e-07 sobre 30 imágenes de VALIDACIÓN (las 20 primeras + 10 defectuosas; tolerancia 1e-3). Celda de ejemplo (validación): `train/12329.png` (buena) → REVISAR, p = 0.381; `train/10905.png` (defectuosa) → DEFECTUOSA, p = 1.000, Grad-CAM sobre la raya; ≈ 0.2 s por imagen en CPU.

**Prueba del Space público (2026-11-13, Eugenia):** con `train/12329.png` (validación, buena) el Space mostró OK, p = 0.281, cuando el modelo oficial da 0.381 (REVISAR). Causa: el redimensionado del navegador no es el de torchvision (D-050). Probabilidad de la misma imagen según el método de redimensionado (exploratorio): bilineal con antialias 0.381 · bicúbico 0.290 · box 0.283 · vecino más cercano 0.286 · Lanczos 0.223. Corregido en `web/index.html` (bilineal con antialias hecho a mano): en 38 imágenes la diferencia con el modelo oficial es como máximo 0.007 (12329 → 0.385, REVISAR). Con la versión vieja, `12160.png` (defectuosa) dio DEFECTUOSA y `10811.png` (buena) dio OK, como se esperaba: el error solo cambia el veredicto en casos dudosos. **Pendiente:** volver a subir `index.html` al Space y repetir la prueba con `12160.png` (defectuosa, esperado ≈ 1.00), `10811.png` (buena, ≈ 0.06) y `12329.png` (≈ 0.38).

## Comparación con el paper del dataset (2026-11-13, valores copiados del paper)
Fuente: Božič, Tabernik y Skočaj (2021), Computers in Industry 129, 103459 (arXiv 2104.06064), sección 5.3 y Fig. 9. El paper **no trae tabla** para KolektorSDD2: los resultados están en la Fig. 9 y el texto cita solo tres valores, que se copian tal cual (los de N = 53 y N = 126 no se citan y no se leen del gráfico).

| Fuente | Método | Supervisión | AP en test |
|---|---|---|---|
| Paper | Segmentación + clasificación end-to-end | Débil, solo etiqueta por imagen (N = 0) | 73,3 % |
| Paper | ídem | Mixta, + 16 imágenes con máscara (N = 16) | 83,2 % |
| Paper | ídem | Completa, todas las máscaras (N = Nall) | 95,4 % |
| Nuestro | CNN ResNet18 fine-tuning (3 semillas) | Solo etiqueta por imagen | 91,6 % ± 0,2 |
| Nuestro | Baseline ResNet18 + logística | Solo etiqueta por imagen | 78,0 % |
| Nuestro | Autoencoder (solo buenas) | Sin etiquetas de defecto | 74,4 % ± 1,5 |

- Igual: split oficial de test (894 + 110) y métrica (AP por imagen).
- Distinto: el paper no menciona validación ni cantidad de corridas; no indica la resolución de entrada para KSDD2 (describe ~230 × 630 y mide velocidad a 232 × 640; nosotros 224 × 608); arquitectura y entrenamiento distintos (50 épocas, batch 1; nosotros ResNet18 preentrenada en ImageNet).
- Lectura: nuestra CNN usa la supervisión del caso débil (N = 0) del paper y supera ese AP (91,6 % vs 73,3 %), pero queda debajo del modelo con todas las máscaras (95,4 %). No es una comparación controlada (cambian arquitectura y preentrenamiento): sirve para ubicar el resultado, no para afirmar que el método es mejor. El paper no reporta F2, recall, FP ni FN para KSDD2.

## Revisión manual de errores de la CNN (2026-11-13, exploratoria)
Análisis **exploratorio** sobre el test ya evaluado (no se usó para decidir nada ni cambia ninguna cifra oficial). Se recalcularon las probabilidades de `cnn_resnet18.pt` (semilla 42) sobre las 8 imágenes y coinciden exactamente con `results/test_scores.npz`. Para cada una se miró la imagen original, una versión con contraste realzado (percentiles 1–99,8), la máscara y el Grad-CAM. Figuras: `docs/img/revision_fp_gradcam.png`, `docs/img/revision_fn_gradcam.png`, `docs/img/revision_fp_vs_defectos_chicos.png`.

**Falsos positivos de mayor probabilidad (máscara vacía):**
| Imagen | Prob. CNN | Qué se ve donde mira el Grad-CAM | Veredicto |
|---|---|---|---|
| 20245 | 0.98 | Punto brillante rojizo de pocos píxeles (≈ 27 px por encima del fondo), justo en el pico del mapa. | Ambiguo: posible defecto mínimo sin anotar |
| 20651 | 0.98 | Mancha tenue azulada e irregular en el borde derecho; la imagen es más oscura que el resto (brillo medio 27 vs ≈ 45). | Ambiguo: posible mancha sin anotar |
| 20002 | 0.93 | Punto blanco saturado (255) en el borde derecho. | Ambiguo: posible defecto mínimo sin anotar |
| 20846 | 0.91 | Punto blanco saturado (255) en el centro-derecha; segundo pico en el borde izquierdo sin nada visible. | Ambiguo: posible defecto mínimo sin anotar |

Los cuatro muestran algo real en la imagen: puntos brillantes compactos, del mismo tipo que los defectos más chicos que sí están anotados (20228 de 87 px, 20231 de 178 px y 20275 de 228 px son grupos de puntos brillantes), pero más chicos. No son errores "en el vacío": la red marca anomalías visibles. Sin el criterio de anotación de Kolektor no se puede decidir si son defectos sin anotar (ruido de etiqueta) o polvo o reflejos aceptables. En el informe se presentan como **casos ambiguos**, no como prueba de ruido de etiquetas.

**Falsos negativos grandes (≥ 3000 px):**
| Imagen | Defecto | Prob. CNN | Contraste vs entorno* | Qué se ve | Grad-CAM | Veredicto |
|---|---|---|---|---|---|---|
| 20141 | 3268 px | 0.09 | −5,1 (más oscuro) | Zona oscura con estrías verticales finas, pegada al borde inferior. | Cae sobre el defecto (22 % de la energía en una máscara que ocupa el 2,3 % de la imagen); el pico queda al lado. | Falla real: defecto visible pero de bajo contraste |
| 20625 | 3271 px | 0.20 | −9,1 (más oscuro) | Zona oscura con estrías en la esquina inferior izquierda. | Pico dentro del defecto, pero hay otra zona igual de fuerte arriba, sin defecto. | Falla real: defecto oscuro y sutil |
| 20969 | 3390 px | 0.45 | +0,4 (casi nulo) | Raya diagonal casi invisible en la original; en la versión realzada se ve como una banda de textura distinta. | Cae sobre el defecto (15 % de la energía vs 2,4 % de la imagen). | Defecto casi invisible; en la demo caería en REVISAR |
| 20772 | 4250 px | 0.17 | +3,9 | Grupo de puntos brillantes dispersos cerca del borde superior derecho (los picos sí son brillantes: p99 108 vs 69 del resto). | Pico dentro del defecto (36 % de la energía vs 2,9 %). | Falla real: defecto visible a ojo |

\* Brillo medio dentro de la máscara menos el de un anillo de 15 px alrededor (escala 0–255).

- **Ninguno es un error de etiqueta evidente:** en los cuatro el defecto se ve (en 20969 solo con contraste realzado).
- **La red mira el lugar correcto pero no alcanza el umbral:** en los cuatro, el Grad-CAM se concentra sobre el defecto. Es la misma conclusión que en los falsos negativos chicos. Ojo: el Grad-CAM se normaliza por imagen y siempre marca algo, así que esto muestra hacia dónde mira la red, no que "haya detectado" el defecto.
- **Rasgo común:** los cuatro tienen poco contraste con el entorno (|contraste| < 10). Cómo se ve en los 110 defectuosos del test, para defectos ≥ 1000 px:

| Grupo | Recall |
|---|---|
| \|contraste\| < 5 | 15/19 (79 %) |
| \|contraste\| ≥ 5 | 62/67 (93 %) |

  Es una tendencia, no una prueba (n chico).
- **Hipótesis descartadas** al mirar los 110 defectuosos del test. Las sugería la inspección visual, pero los números no las sostienen:

| Hipótesis | Recall a favor | Recall en contra |
|---|---|---|
| "Falla en defectos que tocan el borde" | Tocan el borde: 53/63 (84 %) | No lo tocan: 35/47 (74 %) |
| "Falla en defectos más oscuros que el fondo" | Oscuros: 18/24 (75 %) | Claros: 70/86 (81 %) |

- **Zona REVISAR de la demo** (probabilidad entre 0,30 y 0,663, D-048; dato informativo, no se usa para cambiar nada): incluye 8 de los 22 falsos negativos, entre ellos 20969, y 15 piezas buenas. Con un inspector humano revisando esa zona, quedarían 14 defectuosas sin revisar en lugar de 22.


## Pendiente
Fase 7 (informe, presentación). Hechos el 2026-11-13: comparación con el paper, revisión manual de errores (arriba) y README de portfolio; el repo ya es público.
