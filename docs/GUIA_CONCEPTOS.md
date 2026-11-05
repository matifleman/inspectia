# InspectIA — Conceptos desde cero (qué es cada cosa y para qué sirve)

Formato: concepto → analogía → en InspectIA. Los números son los reales de nuestra validación.

## 1. La idea base: aprender de ejemplos
- **Concepto**: en vez de programar reglas ("si hay una línea oscura de más de 3 px, es defecto"), le mostramos ejemplos con la respuesta correcta y el programa descubre solo el patrón. Eso es *machine learning*.
- **Piezas**: *X* = los datos de entrada (la foto), *y* = la respuesta correcta (buena=0 / defectuosa=1, se llama *etiqueta*), *modelo* = lo que aprende a pasar de X a y.
- **Train / validación / test** (analogía del estudio):
  - *Train* = tus apuntes: con esto estudiás.
  - *Validación* = el simulacro: ves cómo vas y ajustás cómo estudiás.
  - *Test* = el examen final: se rinde UNA vez. Si lo mirás antes para cambiar tu forma de estudiar, la nota deja de ser honesta (*fuga de datos*).
- **En InspectIA**: train 1864 fotos, validación 467, test 1004.

## 2. Una imagen es una tabla de números
Una foto en gris de 224×608 (ancho×alto, vertical) son 136.192 números (0 a 255, el brillo de cada píxel). Un modelo no "ve": recibe números. El trabajo es convertir esa tabla gigante en pocos números que describan lo importante. Eso es **extraer features** (*características*).

## 3. HOG — "describir los bordes"
- **Qué es**: *Histogram of Oriented Gradients*. Divide la imagen en cuadraditos de 16×16 píxeles y, en cada uno, anota hacia qué direcciones apuntan los bordes (horizontal, vertical, diagonales…).
- **Analogía**: describirle una cara a alguien por teléfono diciendo, cuadradito por cuadradito, "acá hay una línea vertical, acá una diagonal".
- **Para qué sirve**: captura formas y texturas sin necesidad de entrenar nada. Era el estándar antes de las redes neuronales.
- **Su límite**: no entiende qué es una pieza ni qué es un defecto; solo lista bordes. Si la textura normal de la pieza varía un poco, cambian miles de números. Acá da 17.316 números por foto con solo 197 defectuosas para aprender: demasiado ruido, poca señal. Por eso anduvo mal (AUC 0.74).

## 4. Red neuronal, CNN y ResNet18
- **Red neuronal**: muchas "capas" de operaciones simples encadenadas. Cada capa transforma los números de la anterior. Lo que hace cada operación (sus *pesos*) se ajusta con ejemplos.
- **CNN (red convolucional)**: especializada en imágenes. Usa *filtros* chicos que recorren la foto buscando un patrón. Las primeras capas detectan bordes; las del medio, texturas y formas; las profundas, objetos o patrones complejos (un rayón, una mancha).
- **ResNet18**: una CNN famosa de 18 capas. Su truco son *conexiones residuales* (atajos que suman la entrada de un bloque a su salida), que permiten apilar capas sin que el aprendizaje se degrade.
- **ImageNet**: base de 1,2 millones de fotos de 1000 categorías (perros, autos, tazas…). ResNet18 ya fue entrenada ahí: *preentrenada*.
- **Transfer learning** (analogía): contratar a alguien que ya sabe ver y enseñarle solo tu trabajo, en vez de criar a un bebé desde cero. Con solo 246 defectuosas, entrenar desde cero sobreajustaría; reutilizar lo que ya sabe ver es lo sensato.
- **Congelada vs fine-tuning**: *congelada* = no se toca, se usa como "ojos" que devuelven 512 números por foto (**eso hicimos en la Fase 1**). *Fine-tuning* = se reentrena total o parcialmente con nuestras fotos (**eso viene en la Fase 2**).
- **`model.fc = Identity()`**: la última capa de ResNet decide entre las 1000 categorías de ImageNet; la anulamos para quedarnos con el vector de 512 números anterior.

## 5. Los clasificadores simples (Fase 1)
Toman los números (HOG o ResNet) y deciden buena/defectuosa.
- **Regresión logística**: asigna un peso a cada feature, suma todo y lo convierte en una probabilidad entre 0 y 1 con la función *sigmoide*. Es un "promedio ponderado" que aprende los pesos. Simple, rápido, difícil de sobreajustar.
- **kNN (k vecinos más cercanos)**: para clasificar una foto nueva, busca las *k* fotos de entrenamiento más parecidas y vota. Analogía: "dime con quién andas y te diré quién eres". No aprende nada: memoriza todo. Con 512 o 17.000 dimensiones, "parecido" pierde sentido (*maldición de la dimensionalidad*) y funciona peor.
- **StandardScaler**: pone todas las features en la misma escala (media 0, desvío 1). Sin eso, una feature con números grandes domina las demás.

## 6. Parámetros vs hiperparámetros
- **Parámetros**: lo que el modelo APRENDE solo (los pesos de la logística).
- **Hiperparámetros**: lo que ELEGÍS vos antes de entrenar. Analogía de cocina: los ingredientes se transforman solos en el horno (parámetros), pero la temperatura y el tiempo los elegís vos (hiperparámetros).
- **Los de la Fase 1**:
  - `C` (logística): qué tanto se permite al modelo ajustarse a los datos. C chico = modelo más "conservador" (*regularización* fuerte); C grande = más libre.
  - `k` (kNN): cuántos vecinos votan.
- **Los de la Fase 2 (adelanto)**: *learning rate* (tamaño del paso con que corrige errores: muy alto se pasa de largo, muy bajo no avanza), *batch size* (cuántas fotos mira antes de corregir), *épocas* (cuántas pasadas completas por los datos), *optimizador* (el algoritmo de corrección, p. ej. Adam).
- **Cómo se elige un hiperparámetro**: probás varios valores, medís en VALIDACIÓN, te quedás con el mejor. Nunca en test.

## 7. Sobreajuste (overfitting)
- **Qué es**: el modelo memoriza los ejemplos de entrenamiento en vez de aprender el patrón general. Analogía: estudiar de memoria el examen del año pasado: sacás 10 en ese, pero fallás con preguntas nuevas.
- **Cómo se ve**: error bajo en train, alto en validación.
- **Defensas**: más datos, *regularización* (C chico), *augmentation* (variar las fotos de entrenamiento con flips, rotaciones y brillo), *early stopping* (cortar cuando la validación deja de mejorar), modelos más simples.

## 8. Desbalance de clases
Solo 11% de las fotos son defectuosas. Un modelo perezoso que diga siempre "OK" acierta 89%. Se compensa dándole más importancia a las defectuosas (`class_weight="balanced"` en logística; `pos_weight` en la red) y mirando las métricas correctas.

## 9. Métricas (la parte que más se defiende)
Con el baseline ResNet18+logística en validación (49 defectuosas, 418 buenas):

| | Realidad: defectuosa | Realidad: buena |
|---|---|---|
| **Dijo: defectuosa** | 47 (TP, acierto) | 126 (FP, falsa alarma) |
| **Dijo: buena** | 2 (FN, defecto que se escapa) | 292 (TN, acierto) |

- **Recall** (sensibilidad) = TP / (TP+FN) = 47/49 = 0,96. "De los defectos reales, ¿cuántos agarré?" Es lo que más nos importa.
- **Precision** = TP / (TP+FP) = 47/173 = 0,27. "De lo que marqué, ¿cuánto era defecto de verdad?" Bajo: hay muchas falsas alarmas.
- **F2**: combina las dos, pero el recall pesa 4 veces más que la precisión (por eso 2, de "beta=2"). Nos importa más no dejar escapar defectos que evitar falsas alarmas.
- **Accuracy** = aciertos totales. Engaña con desbalance (ver §8).
- **AUC-ROC** (0.5 = azar, 1 = perfecto): la probabilidad de que una pieza defectuosa elegida al azar tenga un score más alto que una buena elegida al azar. No depende del umbral.
- **AP (Average Precision)**: resume la curva precisión-recall en un número. Más exigente que el AUC cuando hay pocos positivos. Lo usamos para elegir hiperparámetros.
- **Por qué tanto lío con el costo de errores**: un defecto que se escapa llega al cliente (caro); una falsa alarma solo cuesta que un humano revise la pieza (barato). El diseño completo sale de esa asimetría.

## 10. Score y umbral
El modelo no dice "defectuosa" sino un *score* (probabilidad). El **umbral** es la línea que decide: score ≥ umbral → defectuosa. Bajar el umbral = marcás más piezas (menos defectos perdidos, más falsas alarmas). Subirlo = al revés. Elegimos el umbral más alto que todavía agarre el 95% de los defectos, **usando solo validación**.

## 11. Fuga de datos, estratificación y semilla
- **Fuga (data leakage)**: que información del test o de la respuesta se cuele en el entrenamiento y haga que el modelo parezca mejor de lo que es. Ejemplos que evitamos: elegir umbral con el test, darle la máscara al modelo, tener fotos duplicadas entre train y val (casi nos pasa con los `(copy)` de Finder).
- **Estratificar**: al repartir train/validación, mantener la misma proporción de defectuosas (10.6%) en ambos. Sin eso, el azar podría dejarte una validación con casi ninguna.
- **Semilla (seed)**: fija el "azar" para que cualquiera obtenga exactamente los mismos resultados. Es reproducibilidad.

## 12. Normalización de imagen (ImageNet)
ResNet18 se entrenó con fotos restándoles una media y dividiendo por un desvío específicos. Si le damos fotos sin ese preprocesamiento, "ve" cosas raras. Por eso `Normalize(IMAGENET_MEAN, IMAGENET_STD)`.

## 13. Lo que viene (adelanto, para ir ubicándote)
- **Fine-tuning de la CNN (Fase 2)**: reentrenar parte de ResNet18 con nuestras fotos. *Función de pérdida* (BCE): mide qué tan mal predice; el entrenamiento busca minimizarla mediante *descenso por gradiente*.
- **Autoencoder (Fase 3)**: red que comprime y reconstruye la foto. Se entrena SOLO con piezas buenas, así que reconstruye bien lo normal y mal lo raro: el error de reconstrucción funciona como detector de anomalías. Su ventaja: puede detectar defectos nunca vistos.
- **Grad-CAM (Fase 4)**: mapa de calor que muestra qué zona de la foto miró la red para decidir. Sirve para explicar y para ver si mira el defecto o algo irrelevante.

## Glosario relámpago
Feature = número que describe la foto · Etiqueta = respuesta correcta · Peso/parámetro = lo que aprende el modelo · Hiperparámetro = lo que elegís vos · Época = una pasada por todos los datos · Batch = grupo de fotos procesado junto · Overfitting = memorizar · Baseline = modelo simple de referencia que hay que superar · Score = número que da el modelo antes del umbral · Pipeline = cadena de pasos (escalar + clasificar).
