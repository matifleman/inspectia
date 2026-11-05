# InspectIA — Guía de estudio del código (Fases 0 a 2)

> Actualizada tras la migración al repo (D-022): los módulos están en `src/inspectia/` y el umbral oficial es el que maximiza F2 en validación (D-024); `choose_threshold` ahora se llama `threshold_at_recall` y es secundario. Los nombres de funciones conservan la lógica descrita acá; algunos cambiaron al inglés (`make_splits`, `build_index`, `get_transforms`, `best_f2_threshold`...).

Objetivo: que puedas explicar CADA decisión del proyecto sin mirar el código. No hace falta memorizar sintaxis: hay que entender el POR QUÉ.

## 0. Qué hicimos, en 8 líneas
1. Tenemos 3335 fotos de piezas (11% defectuosas) con una máscara que marca el defecto.
2. Auditamos: contamos, buscamos faltantes, miramos tamaños (`audit`).
3. Separamos 20% del train como **validación** (el test no se toca).
4. Sacamos "números" de cada foto de dos formas: **HOG** (clásico) y **ResNet18 preentrenada y congelada** (512 números por foto).
5. Entrenamos clasificadores simples (logística, kNN) sobre esos números.
6. Elegimos hiperparámetros con la validación (por AP) y un **umbral conservador** (recall ≥ 95%).
7. Medimos con F2, recall, AUC, AP (nunca accuracy).
8. Resultado: ResNet18+logística es el baseline útil; los otros no le ganan al "marcar todo".

## 1. Orden de lectura (≈ 60-75 min en total)
| # | Archivo | Tiempo | Qué buscar |
|---|---|---|---|
| 1 | `notebooks/02_baseline.ipynb` | 10 min | El FLUJO completo. Leé los comentarios, no los detalles. |
| 2 | `src/inspectia/data.py` | 20 min | Splits, fuga de datos, transforms. Es lo más importante para defender. |
| 3 | `src/inspectia/evaluate.py` | 15 min | F2, umbral, por qué no accuracy. |
| 4 | `src/inspectia/features.py` | 10 min | Qué es "extraer features" con una red congelada. |
| 5 | `src/inspectia/models.py` (sección baseline) | 10 min | Pipeline, grilla, por qué AP. |
Lo demás (`docs/`, `requirements.txt`, `.gitignore`) no hace falta leerlo para entender el modelo.

## 2. `src/inspectia/data.py` bloque por bloque
- `SEED = 42`, `set_seed()`: fija las semillas de `random`, NumPy y PyTorch (CPU y GPU) para que el split y el entrenamiento den lo mismo cada vez. Reproducibilidad = regla del proyecto. (`PYTHONHASHSEED` fijado en runtime solo afecta a procesos hijos; no es lo que da la reproducibilidad acá.)
- `IMAGENET_MEAN/STD`: media y desvío por canal con que se entrenó ResNet18. La red espera entradas normalizadas así; si no, las features salen peores.
- `DEFAULT_SIZE = (608, 224)`: (alto, ancho). Las fotos son **verticales** (~228 de ancho × ~635 de alto), así que 608×224 mantiene la proporción y no pierde defectos chicos. Ojo: PyTorch `Resize` usa (alto, ancho) y PIL `.size`/`.resize` usan (ancho, alto). Este fue justamente el error que cometimos al principio (ver D-020).
- `build_index(root)`: recorre `train/` y `test/`; ignora las máscaras (`_GT`) y archivos con nombre no numérico (los duplicados de Finder); para cada imagen busca su máscara `<id>_GT.png` y define `label = 1` si la máscara tiene algún píxel > 0. Devuelve un DataFrame: una fila por imagen. **Por qué la etiqueta sale de la máscara**: el dataset no trae un CSV de etiquetas.
- `audit(df)`: `pd.crosstab` cuenta split × clase; saca % de defectuosas; cuenta imágenes sin máscara; abre 50 al azar para ver tamaños y modo de color. Es la Práctica 3 hecha código.
- `make_splits(df, val_frac=0.2)`: el test es el oficial, intocable. De `train` saca 20% para val con `train_test_split(..., stratify=label)`. **Estratificar** = que train y val mantengan el mismo ~10.6% de defectuosas; sin eso, con tan pocas defectuosas, un split podría quedar desparejo por azar.
- `get_transforms(train)`: `ToImage` → `Resize` → (solo si `train=True`: flip horizontal/vertical, rotación ±5°, brillo/contraste ±20%) → `ToDtype(float32, scale=True)` (pasa de 0-255 a 0-1) → `Normalize` (ImageNet). **Augmentation solo en train**: en val/test queremos medir sobre imágenes reales, sin alterar.
- `InspectDataset`: lo que PyTorch necesita para leer datos. `__getitem__` abre la imagen, la pasa a RGB (ResNet espera 3 canales) y devuelve `(imagen, etiqueta)`. **No devuelve la máscara**: si el modelo viera la máscara, tendría la respuesta (fuga). La etiqueta es `float32` porque la pérdida BCE lo pide.
- `load_mask`: carga la máscara aparte, solo para evaluar localización después.
- `pos_weight(df)` = n_buenas / n_defectuosas (≈ 8.5 en train). Se usará en la Fase 2: `BCEWithLogitsLoss(pos_weight=...)` hace que equivocarse en una defectuosa pese ~8.5 veces más.
- `make_loaders`: arma los `DataLoader` (lotes, mezcla en train, generador con semilla). Todavía NO se usa en baselines; lo usa la Fase 2.

## 3. `src/inspectia/features.py`
- `get_device()`: CUDA (NVIDIA) > MPS (GPU de Mac) > CPU.
- `resnet_features(df)`:
  - `@torch.no_grad()`: no calcula gradientes (no vamos a entrenar): menos memoria y más rápido.
  - `resnet18(weights=DEFAULT)`: red ya entrenada en ImageNet (1.2M de fotos). **Transfer learning**: reutilizamos lo que ya sabe ver (bordes, texturas).
  - `model.fc = Identity()`: la última capa (1000 clases de ImageNet) se anula; la salida pasa a ser el vector de **512 números** tras el pooling global. Esas 512 cifras son la "descripción" de la imagen.
  - `model.eval()`: apaga comportamientos de entrenamiento (BatchNorm usa estadísticas fijas).
  - `shuffle=False` en el DataLoader: **crítico**. Las features tienen que quedar en el mismo orden que las etiquetas de `df`.
  - Usa `get_transforms(False)`: sin augmentation, porque acá solo medimos.
- `hog_features(df)`: HOG = Histogram of Oriented Gradients, cuenta direcciones de bordes por celda. Imagen en gris, 608 de alto × 224 de ancho, celdas de 16×16 → 38×14 celdas; bloques de 2×2 celdas con 9 orientaciones → (13×37) bloques × 36 = **17.316 dimensiones** (coincide con la salida del notebook). Es el método "clásico" visto en la cátedra (C3A).

## 4. `src/inspectia/models.py` (sección baseline)
- `GRIDS`: valores a probar: C (logística) y k (kNN).
- `_build`: `make_pipeline(StandardScaler(), clasificador)`. El scaler deja cada feature con media 0 y desvío 1; kNN usa distancias y la logística regulariza, y ambos son sensibles a la escala. Dentro del pipeline el scaler se ajusta **solo con train** → no hay fuga.
  - Logística: `C` = inverso de la regularización (C chico = modelo más simple); `class_weight="balanced"` pondera las clases según su frecuencia para compensar el 11%.
  - kNN: `weights="distance"`: los vecinos más cercanos pesan más → scores menos "escalonados".
- `select_model`: para cada valor de la grilla, entrena en train, predice probabilidades en val, calcula **Average Precision** y se queda con el mejor. Se usa AP porque no depende del umbral (el umbral se elige después) y es estable con clases desbalanceadas.

## 5. `src/inspectia/evaluate.py`
- `evaluate(y, scores, thr)`: `pred = score >= thr`; matriz de confusión (`tn, fp, fn, tp`); F2, recall, precision, AUC, AP.
- **F2** = 5·P·R / (4·P + R): como F1 pero el recall pesa 4 veces lo que la precisión. Coherente con "dejar pasar un defecto sale caro".
- `best_f2_threshold(y_val, s_val)` (oficial) y `threshold_at_recall(y_val, s_val, 0.95)` (secundario): con `precision_recall_curve` obtiene recall para cada umbral posible; se queda con los umbrales con recall ≥ 0.95 y elige el **más alto** (menos falsas alarmas manteniendo el recall). Si ninguno llega, usa el que maximiza F2. **Límite conocido (D-018)**: con scores pobres o discretos cae a umbral 0 y marca todo.
- `best_f2_threshold`: prueba cada score posible como umbral y se queda con el de mayor F2.
- `always_ok_baseline`: referencias absurdas pero necesarias. "Siempre OK": accuracy 89.5% con recall 0 → muestra por qué accuracy engaña. "Marcar todo": recall 1.0 pero F2 ≈ 0.37 → la vara mínima.
- `plot_curves`: curvas ROC y Precision-Recall de varios modelos juntos.

## 6. Conceptos que tenés que poder explicar (con números de ESTE proyecto)
- **Fuga de datos**: si el test influye en cualquier decisión (hiperparámetro, umbral), su resultado ya no es honesto. Por eso: val aparte, test una sola vez.
- **Accuracy engaña**: "siempre OK" = 89.5% de acierto y detecta 0 defectos.
- **Precision vs recall**: recall = de las 49 defectuosas, cuántas detecté (47); precision = de las que marqué, cuántas lo eran (27%).
- **ROC vs PR**: con 11% de positivos, PR es más exigente y más informativa. AUC 0.948 suena enorme; AP 0.868 y precisión 0.27 muestran el costo real (126 falsas alarmas).
- **Umbral**: el modelo da un score; el umbral decide "defectuosa o no". Subirlo = menos falsas alarmas, más defectos perdidos.
- **Features congeladas vs fine-tuning**: ahora la red no aprende; solo la logística. En la Fase 2 la red se reentrena parcialmente.
- **Ruido con 49 positivos**: cada defectuosa = 2 puntos de recall. Los números de val son indicativos, no exactos.

## 7. Preguntas que te pueden hacer (practicá la respuesta en voz alta)
1. ¿Por qué no usaron accuracy? ¿Qué pasa con un clasificador que siempre dice OK?
2. ¿Cómo evitaron fuga de datos? ¿Quién eligió el umbral y con qué datos?
3. ¿Por qué estratificar la validación?
4. ¿Por qué F2 y no F1?
5. ¿Qué hace `model.fc = Identity()` y por qué sirve?
6. ¿Por qué el kNN y HOG funcionan peor? ¿Qué significa que no superen a "marcar todo"?
7. ¿Por qué elegir hiperparámetros por AP y no por F2?
8. ¿Qué límite tiene reportar recall con 49 defectuosas?
9. ¿Por qué no se normaliza distinto en val y test?
10. ¿Qué harían si el test diera muy distinto de val?

## 8. Ejercicio para entender de verdad (20 min, opcional)
En el notebook, después de la celda 5: tomá los scores de validación de ResNet18+logística y, a mano con `evaluate(y_va, scores, thr)`, probá thr = 0.1, 0.2, 0.5, 0.8. Anotá cómo cambian FN y FP. Eso es el trade-off del umbral, y es lo que vas a explicar en la Práctica 5.
