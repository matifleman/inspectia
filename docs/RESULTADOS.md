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

## Pendiente
3 semillas de la CNN final (y de scratch, para ver su varianza), Fase 4 (autoencoder), Fase 5 (test).
