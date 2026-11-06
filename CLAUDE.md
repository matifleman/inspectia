# InspectIA — instrucciones para agentes de IA (Claude Code, Cursor, Codex, etc.)

Proyecto final de IA (grupo de 4, UCA): inspección visual de defectos superficiales en KolektorSDD2.
CNN supervisada (ResNet18) vs autoencoder (anomalías) vs baseline (features ResNet18 + logística).
**Entrega: lunes 16/11/2026, con TODO el plan (Fases 0–7), sin recortes (D-023).**

## Antes de hacer cualquier cosa
1. Leé `docs/ESTADO.md` (foto actual, mapa del código, qué falta y quién lo hace).
2. Si vas a tocar un experimento o una decisión, leé `docs/DECISIONES.md` (log append-only) y `docs/RESULTADOS.md` (solo números reales).
3. No contradigas decisiones vigentes sin avisar al usuario. Si algo cambia, agregá una entrada nueva en DECISIONES.md que reemplace a la vieja.

## Reglas no negociables
- Split oficial de KolektorSDD2 (`splits/*.csv`, semilla 42). Validación = 20 % estratificado del train.
- **El test NUNCA se usa para elegir hiperparámetros, umbral, época ni modelo.** Ya se evaluó UNA vez (Fase 5, D-045/D-047). No lo vuelvas a puntuar para decidir nada; todo análisis posterior es exploratorio y se declara así. No borres ni saltees el lock `results/test_scores.npz` / `TEST_EVALUADO.txt`.
- La máscara del defecto NO es entrada de ningún clasificador (solo para visualizar/evaluar Grad-CAM).
- Augmentation solo en train.
- Nunca accuracy como métrica principal (~89 % de las imágenes son buenas): F2, recall de defectuosas, AUC-ROC, AP.
- Umbral = máx. F2 en validación, fijo. Un falso negativo cuesta mucho más que un falso positivo.
- Reproducibilidad: semilla fija, versiones en `requirements.txt`, un notebook por fase.
- El dataset NO va al repo (CC BY-NC-SA 4.0: se cita, no se redistribuye).

## Honestidad
- No inventes números ni resultados. Si algo está PENDIENTE (p. ej. comparación con el paper), pedí los datos reales.
- Si ves fuga de datos, sobreajuste, diseño flojo o métrica mal usada, decilo directamente.

## Cómo trabajar
- Idioma: español rioplatense. Los integrantes son de 5.º año: no expliques lo básico de más.
- Stack: Python 3.11+, PyTorch + torchvision, scikit-learn, ResNet18, Grad-CAM (implementado a mano en `explain.py`), Weights & Biases, Gradio.
- La lógica va en módulos `src/inspectia/*.py`; los notebooks (Colab) solo corren experimentos.
- Config por YAML en `configs/`. Un cambio por experimento.
- Todo lo hecho o decidido se documenta en `docs/` (ESTADO, DECISIONES, RESULTADOS) para poder retomar sin leer todo el código. Al cerrar una sesión, actualizá ESTADO.md.
- Comandos git: no hagas push sin que el usuario lo pida; trabajá en rama propia y PR a `main` (revisa Mati).

## Artefactos que NO están en git
`checkpoints/`, `results/`, `features/`, `data/` están en `.gitignore`. Se comparten por la carpeta de Drive `InspectIA/` (ver ESTADO.md, sección "Artefactos"). Los números oficiales están copiados en `docs/RESULTADOS.md`.
