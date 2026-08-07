# AI Vision Suite: Clasificación de Mascotas y OCT Retinal con CNN + LLM

**Maestría en Ciencia de Datos — Aprendizaje Profundo**
**Autor:** Oscar Ferreira

## 1. Objetivo

Construir y desplegar un sistema de clasificación de imágenes basado en redes
neuronales convolucionales (CNN), aplicado a dos dominios distintos:

1. **Mascotas** — identificación de especie (perro/gato) y raza a partir de una
   foto.
2. **OCT retinal** — clasificación de imágenes de tomografía de coherencia óptica
   en 4 categorías clínicas (CNV, DME, DRUSEN, NORMAL).

Ambos módulos se complementan con una capa de interpretación en lenguaje natural
mediante GPT-4o-mini, e integran un sistema desplegado y accesible públicamente:

**Demo:** https://huggingface.co/spaces/Ferreira91/Caso-Negocio-CNN

## 2. Datasets

### 2.1 Oxford-IIIT Pet Dataset (módulo mascotas)

Reorganizado en dos carpetas raíz (`PERROS` y `GATOS`), cada una con subcarpetas
por raza:

| Grupo  | Imágenes | Razas (clases) |
|--------|---------:|---------------:|
| Perros | 4,990    | 25             |
| Gatos  | 2,400    | 12             |
| **Total** | **7,390** | **37**      |

El dataset está desbalanceado entre perros y gatos (67% / 33%).

### 2.2 Dataset OCT retinal (módulo OCT)

Imágenes en escala de grises (160×160) organizadas en 4 clases clínicas: **CNV**
(neovascularización coroidea), **DME** (edema macular diabético), **DRUSEN** y
**NORMAL**.

## 3. Módulo Mascotas

### 3.1 Fase 1 — Clasificación de especie (perro vs. gato)

Se entrenaron **tres variantes** de una CNN secuencial desde cero, variando la
profundidad, para comparar el efecto de agregar bloques convolucionales:

- Arquitectura común: bloques `Conv2D` + `MaxPooling2D`, `Dropout` antes de la
  capa densa final, `Dense(1, sigmoid)` de salida. Imágenes a 224×224×3.

| Variante | Test accuracy |
|---|---:|
| 3 capas conv | 74.9% |
| 4 capas conv | 83.6% |
| **5 capas conv (seleccionada)** | **88.1%** |

La variante de 5 capas fue la de mejor desempeño y es la que se usa en el sistema
desplegado.

### 3.2 Fase 2a — Transfer learning y fine-tuning: razas de perro (25 clases)

- Base preentrenada **MobileNetV2** (ImageNet), `include_top=False`.
- Head propio: `GlobalAveragePooling2D` → `Dropout` → `Dense(relu)` → `Dropout` →
  `Dense(25, softmax)`.
- Se entrenaron dos versiones: **transfer learning puro** (base congelada) y
  **fine-tuning** (capas superiores de MobileNetV2 descongeladas y reentrenadas
  con tasa de aprendizaje baja).

| Estrategia | Accuracy |
|---|---:|
| Transfer learning | 93% |
| Fine-tuning | 93% |

La mejora del fine-tuning sobre transfer learning fue marginal en accuracy
agregada, aunque se observó mejor separación en clases visualmente similares.
Las razas con mayor confusión fueron **American pitbull** (se confunde con
American bulldog) y **Staffordshire bullterrier**, consistente con el parecido
visual entre razas de la misma familia.

### 3.3 Fase 2b — Transfer learning y fine-tuning: razas de gato (12 clases)

Misma arquitectura y estrategia que 3.2, aplicada al subconjunto de gatos.

| Estrategia | Accuracy |
|---|---:|
| Transfer learning | 87% |
| **Fine-tuning** | **90%** |

Aquí el fine-tuning sí aportó una mejora más notoria (+3 puntos). Las mayores
confusiones se dieron entre **Birman** y **Ragdoll**, y entre **Bengal** y
**Egyptian Mau** — pares de razas con pelaje y patrón muy similares.

## 4. Módulo OCT Retinal

CNN de 5 bloques convolucionales entrenada desde cero (`Conv2D` + `BatchNormalization`
+ `MaxPooling2D` en cada bloque, seguida de `GlobalAveragePooling2D` y una cabeza
densa con `Dropout`), para clasificación en 4 clases.

| Métrica | Valor |
|---|---:|
| Test accuracy | 93.9% |
| F1 macro avg | 0.92 |

Este módulo es educativo/demostrativo: la aplicación desplegada aclara
explícitamente que no sustituye una valoración oftalmológica profesional.

## 5. Arquitectura del sistema desplegado

El sistema en producción (Hugging Face Space, Streamlit) usa **un único modelo
seleccionado por fase** — el de mejor desempeño de cada comparación anterior — en
vez de mantener las variantes exploratorias en paralelo:

- Especie: CNN de 5 capas (Fase 1).
- Raza: modelo de fine-tuning (Fase 2a/2b), por ser la versión de mejor o igual
  desempeño frente a transfer learning puro en ambos casos.
- OCT: CNN de 5 bloques.

**Capa de razonamiento con LLM (GPT-4o-mini):**

- **Mascotas:** el resultado numérico se traduce a lenguaje natural con un
  esquema de 3 modos según umbrales de confianza (SEGURO / DUDA MEDIA / NO
  CONCLUYENTE). El LLM solo recibe la imagen como entrada visual cuando la
  confianza del clasificador es baja; con confianza alta, redacta la explicación
  basándose únicamente en los datos numéricos del modelo.
- **OCT:** se genera siempre una explicación educativa de 8 secciones (resultado,
  causas probables, síntomas asociados, tratamientos generales, nivel de
  confianza, limitaciones, etc.), con reglas explícitas en el prompt para evitar
  diagnósticos definitivos o recomendaciones médicas específicas.

Esta capa de razonamiento es la principal diferencia entre el sistema desplegado
y un clasificador CNN convencional: convierte un score numérico en una
explicación interpretable para un usuario no técnico, y decide de forma
condicionada cuándo vale la pena el costo adicional de que el LLM analice la
imagen directamente.

## 6. Análisis

- **CNN desde cero vs. transfer learning:** para especie (tarea binaria más
  simple), la CNN desde cero alcanza como máximo 88.1% incluso con 5 bloques
  convolucionales. Para razas (tarea de grano fino, 25 y 12 clases), transfer
  learning con MobileNetV2 alcanza 87–93% partiendo de representaciones
  preentrenadas en ImageNet — evidencia de que los filtros preentrenados
  (bordes, texturas, formas) aceleran y mejoran el aprendizaje en dominios con
  relativamente pocos datos por clase, incluso en tareas más difíciles que la
  binaria.
- **Efecto de la profundidad (Fase 1):** el salto de accuracy entre variantes
  (74.9% → 83.6% → 88.1%) es consistente y decreciente en magnitud, sugiriendo
  rendimientos marginales decrecientes al seguir agregando profundidad sin
  cambiar la estrategia de regularización.
- **Transfer learning vs. fine-tuning:** el beneficio del fine-tuning no fue
  uniforme entre especies — marginal en perros (93% → 93%) y más notorio en
  gatos (87% → 90%). Es consistente con que el fine-tuning ayuda más cuando hay
  más margen de mejora sobre el transfer learning puro.
- **Confusión entre clases:** el principal reto en ambos módulos de raza es la
  similitud visual dentro de familias de razas (pitbull/bulldog,
  Birman/Ragdoll, Bengal/Egyptian Mau) — un límite esperable de clasificación de
  grano fino basada solo en apariencia, que ni el fine-tuning elimina por
  completo.

## 7. Limitaciones y trabajo futuro

- **Interpretabilidad visual (Grad-CAM):** no implementado en esta entrega por
  restricción de tiempo. Permitiría visualizar qué regiones de la imagen
  influyeron en cada predicción, más allá del score numérico.
- **Evaluación cuantitativa de las explicaciones del LLM:** no hay actualmente
  una métrica formal de calidad/fidelidad de las explicaciones generadas por
  GPT-4o-mini respecto a la predicción del modelo.
- **Segmentación:** el dataset Oxford-IIIT Pet incluye trimaps de segmentación no
  utilizados en esta versión; el módulo de mascotas podría extenderse a
  segmentación además de clasificación.
- **Desbalance de clases:** algunas razas y la clase DRUSEN en OCT muestran menor
  precision/recall relativo — candidatas a técnicas de balanceo adicionales
  (oversampling, focal loss).

## 8. Conclusiones

1. El transfer learning con una red preentrenada (MobileNetV2) supera de forma
   consistente a una CNN entrenada desde cero, incluso al resolver una tarea más
   difícil (clasificación fina de 25/12 clases vs. binaria).
2. El fine-tuning aporta una mejora adicional sobre el transfer learning puro,
   con impacto distinto según el dominio (mayor en gatos que en perros).
3. Complementar la salida del clasificador con una capa de razonamiento en
   lenguaje natural, condicionada por la confianza del modelo, agrega
   interpretabilidad sin incurrir en el costo de invocar al LLM con imagen en
   cada predicción.
4. El principal reto pendiente sigue siendo la confusión entre clases visual y
   semánticamente similares, así como la falta de interpretabilidad visual
   (Grad-CAM) sobre las decisiones del modelo — ambos quedan como trabajo
   futuro.

## 9. Referencias

- Oxford-IIIT Pet Dataset — https://www.robots.ox.ac.uk/~vgg/data/pets/
- Keras Applications — MobileNetV2 (preentrenada en ImageNet)
- Material de clase: CNN, Transfer Learning y Fine-tuning (Alberto Benavides,
  Maestría en Ciencia de Datos)
