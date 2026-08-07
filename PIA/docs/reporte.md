# Clasificación de imágenes con Redes Neuronales Convolucionales

**Maestría en Ciencia de Datos — Aprendizaje Profundo**
**Autor:** Oscar Ferreira

## 1. Objetivo

Construir una aplicación de clasificación de imágenes con redes neuronales
convolucionales (CNN), resuelta en dos etapas y dos notebooks:

1. **`Proyecto_Final.ipynb`** — entrenamiento de los modelos base: una CNN desde
   cero (clasificación binaria) y dos modelos de transfer learning (clasificación
   de razas).
2. **`Demo_Proyecto_V1.ipynb`** — pipeline de inferencia que combina varios modelos
   entrenados (ensemble por confianza) en dos fases sucesivas —especie y luego
   raza— y presenta el resultado en un dashboard visual.

## 2. Dataset

Se usó el **Oxford-IIIT Pet Dataset**, reorganizado en dos carpetas raíz (`PERROS` y
`GATOS`), cada una con subcarpetas por raza:

| Grupo  | Imágenes | Razas (clases) |
|--------|---------:|---------------:|
| Perros | 4,990    | 25             |
| Gatos  | 2,400    | 12             |
| **Total** | **7,390** | **37**      |

El dataset está desbalanceado entre perros y gatos (67% / 33%).

## 3. Notebook 1 — `Proyecto_Final.ipynb`: entrenamiento de los modelos base

### 3.1 Experimento 1 — CNN desde cero: Perro vs. Gato (binaria)

Arquitectura secuencial construida desde cero:

- `data augmentation` (flip horizontal, rotación ±10%, zoom ±10%)
- 3 bloques `Conv2D` + `MaxPooling2D` (32 → 64 → 128 filtros, kernels 3×3)
- `Dropout(0.30)` tras el último bloque convolucional
- `Flatten` → `Dense(128, relu)` → `Dropout(0.50)` → `Dense(1, sigmoid)`
- Optimizador Adam, `binary_crossentropy`, `class_weight` balanceado (para
  compensar el desbalance perro/gato), `EarlyStopping` (paciencia 4)
- Imágenes a 224×224×3, split 80/20 estratificado

**Resultado:** 81.9% test accuracy, 0.384 test loss (20 épocas, sin activar
early stopping).

Matriz de confusión (filas = real, columnas = predicho; orden PERRO, GATO):

```
            PERRO  GATO
PERRO         835   163
GATO          105   375
```

| Clase | Precisión | Recall | F1 |
|---|---:|---:|---:|
| PERRO | 0.89 | 0.84 | 0.86 |
| GATO  | 0.70 | 0.78 | 0.74 |

### 3.2 Experimento 2 — Transfer learning: razas de perro (25 clases)

- Base preentrenada **MobileNetV2** (ImageNet), `include_top=False`, **congelada**
- Head propio: `GlobalAveragePooling2D` → `Dropout(0.30)` → `Dense(128, relu)` →
  `Dropout(0.30)` → `Dense(25, softmax)`
- Mismo pipeline de `data augmentation` y `EarlyStopping`

**Resultado:** 93.3% validation accuracy, 0.228 validation loss (early stopping en
época 15). `macro avg` F1 = 0.93. Las razas con más confusión son **American
pitbull** (recall 0.76, se confunde con American bulldog) y **Staffordshire**
(recall 0.88).

### 3.3 Experimento 3 — Transfer learning: razas de gato (12 clases)

Misma arquitectura y configuración que el experimento 2, aplicada al subconjunto de
gatos.

**Resultado:** 87.5% validation accuracy, 0.349 validation loss (early stopping en
época 11). `macro avg` F1 = 0.87. La mayor confusión ocurre entre **Birman** y
**Ragdoll**, y entre **Bengal** y **Egyptian Mau** — razas de pelaje y patrón
similares.

## 4. Notebook 2 — `Demo_Proyecto_V1.ipynb`: pipeline de inferencia con ensemble

Este notebook no entrena modelos: carga modelos ya entrenados y construye un
pipeline de inferencia en **dos fases**, con un dashboard HTML como salida.

### 4.1 Fase 1 — Detección de especie (ensemble de 3 CNN)

Se cargan tres variantes de CNN entrenadas para la tarea binaria perro/gato, con
distinta profundidad: **3, 4 y 5 capas convolucionales**. Cada una predice de forma
independiente sobre la imagen de entrada (umbral ajustado a 0.37, en vez de 0.5,
para favorecer la detección de la clase minoritaria "gato"). La especie final se
decide por el modelo con **mayor confianza** entre los tres; el número de modelos
que coinciden en la especie ganadora se muestra como referencia visual (no afecta
la decisión).

### 4.2 Fase 2 — Clasificación de raza (Transfer Learning vs. Fine-Tuning)

Según la especie detectada en la Fase 1, se cargan dos modelos de raza para esa
especie: uno de **transfer learning puro** (base MobileNetV2 congelada, igual que
en el notebook 1) y uno de **fine-tuning** (mismo modelo, pero con las capas
superiores de MobileNetV2 descongeladas y reentrenadas). Gana el de mayor
confianza. Adicionalmente se calcula la unión de los Top-5 de ambos modelos para
comparar visualmente dónde coinciden y dónde difieren.

### 4.3 Dashboard de salida

El notebook genera un dashboard HTML con: imagen de prueba, especie y raza
estimadas, modelo ganador de cada fase, tarjetas comparativas de las 3 CNN de
especie y de los 2 modelos de raza (con barra de confianza y badge de "ganador"), y
dos gráficas (Top-5 por modelo, y comparativa de la unión de ambos Top-5).

### 4.4 Ejemplo real de ejecución

Sobre una imagen de prueba, el pipeline produjo:

| Fase 1 — Especie | Predicción | Confianza |
|---|---|---:|
| 3 Capas CNN | Perro | 78.06% |
| 4 Capas CNN | Perro | 94.04% |
| **5 Capas CNN (ganador)** | **Perro** | **97.08%** |

Los 3 modelos coincidieron en la especie (3/3).

| Fase 2 — Raza | Predicción | Confianza |
|---|---|---:|
| Transfer Learning | American pitbull | 93.04% |
| **Fine Tuning (ganador)** | **American pitbull** | **94.21%** |

**Resultado final del pipeline:** Perro, raza American pitbull, 94.21% de
confianza — con ambos modelos de raza coincidiendo en la predicción, lo cual
refuerza la confiabilidad del resultado.

## 5. Análisis

- **CNN desde cero vs. transfer learning (notebook 1):** para un problema binario
  más simple, la CNN desde cero (81.9%) queda por debajo de los modelos de
  transfer learning (93.3% y 87.5%) aun cuando estos resuelven tareas más difíciles
  (25 y 12 clases de grano fino). Los filtros preentrenados en ImageNet aportan
  representaciones generales (bordes, texturas, formas) que aceleran y mejoran el
  aprendizaje en dominios con relativamente pocos datos por clase.
- **Ensemble por profundidad (Fase 1 del demo):** la CNN de 5 capas fue la más
  confiada en el ejemplo analizado (97.08%), seguida por la de 4 capas (94.04%) y
  la de 3 capas (78.06%) — consistente con que arquitecturas más profundas capturan
  representaciones más discriminativas, aunque con mayor riesgo de sobreajuste que
  debe vigilarse con más ejemplos.
- **Transfer learning vs. fine-tuning (Fase 2 del demo):** el fine-tuning (94.21%)
  superó ligeramente al transfer learning puro (93.04%) en el ejemplo evaluado,
  esperable ya que descongelar y reentrenar las capas superiores de MobileNetV2
  permite ajustar representaciones más específicas al dominio de razas de mascota,
  a costa de mayor tiempo de entrenamiento y riesgo de sobreajuste si no se
  regula con una tasa de aprendizaje baja.
- **Valor de la estrategia de ensemble:** usar "gana el de mayor confianza" en
  ambas fases, en vez de un solo modelo fijo, añade robustez: si un modelo da una
  predicción dudosa (confianza baja), el pipeline puede apoyarse en el que esté más
  seguro, en vez de forzar siempre la salida del mismo modelo.

## 6. Conclusiones

1. El transfer learning con una red preentrenada (MobileNetV2) supera de forma
   consistente a una CNN entrenada desde cero, tanto en accuracy como en velocidad
   de convergencia, cuando el dataset es mediano/pequeño.
2. El fine-tuning aporta una mejora adicional sobre el transfer learning puro al
   permitir que el modelo ajuste representaciones más específicas del dominio.
3. Combinar varios modelos en un esquema de ensemble por confianza (en vez de
   depender de un único modelo) da un pipeline de inferencia más robusto,
   especialmente útil en un dashboard de producción/demo donde se necesita una
   respuesta única y confiable por imagen.
4. El principal reto sigue siendo la confusión entre clases visual y
   semánticamente similares (razas de la misma familia), donde ni el fine-tuning
   elimina completamente la ambigüedad.

## 7. Referencias

- Oxford-IIIT Pet Dataset — https://www.robots.ox.ac.uk/~vgg/data/pets/
- Keras Applications — MobileNetV2 (preentrenada en ImageNet)
- Material de clase: CNN, Transfer Learning y Fine-tuning (Alberto Benavides,
  Maestría en Ciencia de Datos)
