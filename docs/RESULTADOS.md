# InspectIA — Resultados reales (solo números con corrida detrás)

Todo en VALIDACIÓN (467 imágenes, 49 defectuosas = 20 % estratificado del train, seed 42, entrada 608×224 correcta).
**TEST: todavía NO evaluado** (se evalúa una vez, en `05_evaluacion`, D-026).
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

## Pendiente
Semillas de scratch (opcional), Fase 5 (test, una sola vez).
