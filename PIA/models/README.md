# Modelos entrenados

Todos los modelos se entrenaron y guardaron originalmente en formato nativo `.keras`. Para compatibilidad con la carga en Hugging Face Spaces se convirtieron a `.h5` / `.weights.h5` antes del despliegue.

| Archivo | Módulo | Arquitectura | Clases | Test accuracy | Notebook de origen |
|---|---|---|---|---|---|
| `5capas_modelo_fase1_perro_gato_TF215.h5` | Mascotas — Fase 1 | CNN secuencial, 5 bloques conv (desde cero) | 2 (perro/gato) | 88.1% | `notebooks/01_fase1_especie_perro_gato.ipynb` |
| `modelo_razas_perros_FineTuning.h5` | Mascotas — Fase 2a | MobileNetV2 (transfer learning + fine-tuning) | 25 razas de perro | 93% | `notebooks/02_fase2_razas_perros.ipynb` |
| `modelo_razas_gatos_FineTuning.h5` | Mascotas — Fase 2b | MobileNetV2 (transfer learning + fine-tuning) | 12 razas de gato | 90% | `notebooks/03_fase2_razas_gatos.ipynb` |
| `pesos_oct.weights.h5` | OCT retinal | CNN secuencial, 5 bloques conv (desde cero) | 4 (CNV/DME/DRUSEN/NORMAL) | 93.9% | `notebooks/04_oct_clasificacion.ipynb` |

## Notas de conversión

- Los modelos de mascotas se guardan completos (arquitectura + pesos) en `.h5`, cargados directamente con `tf.keras.models.load_model(path, compile=False)`.
- El modelo OCT se guarda **solo como pesos** (`.weights.h5`); la arquitectura se reconstruye en código (`app/app.py`) antes de cargar los pesos con `model.load_weights(...)`.
- TensorFlow usado en el entrenamiento original: TF 2.15 (ver sufijo del archivo de Fase 1).

## Tamaño de archivos

> ⚠️ Completar con el peso real de cada `.h5` antes de subir. Si algún archivo supera ~100 MB, GitHub estándar lo rechaza — usar [Git LFS](https://git-lfs.com/) o alojar los pesos únicamente en el Space (Hugging Face no tiene ese límite) y dejar aquí el link de descarga en vez del binario.

## Reproducibilidad

Cada modelo puede reentrenarse desde su notebook correspondiente en `notebooks/`, configurando las variables de entorno `DATA_DIR` (ruta al dataset) y `MODELS_DIR` (ruta de salida de los pesos). Los notebooks conservan los outputs de la ejecución original (curvas de entrenamiento, matrices de confusión, `classification_report`) como evidencia — no es necesario re-ejecutarlos para que la app funcione.
