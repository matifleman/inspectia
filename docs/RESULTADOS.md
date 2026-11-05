# InspectIA — Resultados reales (solo números con corrida detrás)

> ⚠️ **INVALIDADOS (D-020):** la tabla de abajo se calculó con imágenes redimensionadas al revés (224×608 como alto×ancho). Quedan solo como registro histórico; hay que rehacer la Fase 1 con el tamaño correcto (608×224). No citar estos números en el informe.

Cada fila: fecha · fase · modelo · split · métricas · origen. Si no hay corrida, no hay fila.
Todo en VALIDACIÓN (467 imágenes, 49 defectuosas = 20% estratificado del train, seed 42). TEST: todavía NO evaluado.

## Fase 1 — Baselines, validación (2026-11-05, `notebooks/01_baselines.ipynb`, celdas 1-6, local Mac)
| Modelo | Hiperparámetro elegido (por AP en val) | Umbral | F2 | Recall | Precision | AUC-ROC | AP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| ResNet18 + logística | C=0.01 | 0.203 | **0.637** | 0.959 (47/49) | 0.272 | **0.948** | **0.868** | 2 | 126 |
| ResNet18 + kNN | k=21 | 0.000* | 0.370 | 1.000 | 0.105 | 0.807 | 0.617 | 0 | 418 |
| HOG + logística | C=10 | ~0.000* | 0.365 | 0.959 | 0.105 | 0.739 | 0.433 | 2 | 401 |
| HOG + kNN | k=5 | 0.000* | 0.370 | 1.000 | 0.105 | 0.540 | 0.231 | 0 | 418 |

\* Umbral degenerado: marca prácticamente todo como defectuosa.

Referencias triviales en val: "siempre OK" → accuracy 0.895, recall 0, F2 0. "Marcar todo defectuosa" → precision 0.105, recall 1.0, **F2 0.370**.
Referencias en test (calculadas analíticamente, 110/1004): "marcar todo" → precision 0.110, F2 0.381.

Grilla ResNet18+logística (AP val): C=0.001 0.851 · 0.01 0.868 · 0.1 0.856 · 1 0.853 · 10 0.855 (diferencias chicas, dentro del ruido con 49 positivos).
