# AI Vision Suite — Clasificación de Mascotas y OCT Retinal con CNN + LLM

Sistema de visión computacional con dos módulos independientes (clasificación de mascotas y clasificación de imágenes OCT retinales), ambos construidos sobre redes convolucionales y complementados con una capa de interpretación en lenguaje natural mediante GPT-4o-mini con lógica de confianza.

**Demo en vivo:** https://huggingface.co/spaces/Ferreira91/Caso-Negocio-CNN

---

## Contenido

- [Problema de negocio](#problema-de-negocio)
- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Módulo 1: Clasificador de mascotas](#módulo-1-clasificador-de-mascotas)
- [Módulo 2: Clasificador OCT retinal](#módulo-2-clasificador-oct-retinal)
- [Capa de razonamiento con LLM](#capa-de-razonamiento-con-llm)
- [Resultados](#resultados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo correr localmente](#cómo-correr-localmente)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)

---

## Problema de negocio

Los sistemas de clasificación de imágenes basados únicamente en un score de confianza numérico son difíciles de interpretar para un usuario no técnico: un "78% de confianza" no le dice a un usuario final si debe confiar en el resultado o no, ni por qué. Este proyecto explora un patrón de **cascada de modelos + fallback condicionado por confianza hacia un LLM**, donde la red neuronal resuelve la clasificación y un modelo de lenguaje interviene únicamente cuando la certeza del clasificador es baja, generando una explicación en lenguaje natural en vez de solo un número.

El mismo patrón arquitectónico se aplica a dos dominios distintos para probar su generalización:

1. **Mascotas** (caso de negocio / consumer): identificación de especie y raza a partir de una foto.
2. **OCT retinal** (caso educativo / salud): clasificación de imágenes de tomografía de coherencia óptica en 4 categorías clínicas.

## Arquitectura del sistema

```
                    ┌─────────────┐
   Imagen  ───────▶ │  CNN(s)     │
                    └──────┬──────┘
                           │
                 ¿confianza suficiente?
                     /            \
                   Sí              No
                   │                │
                   ▼                ▼
          Respuesta directa   GPT-4o-mini recibe
          (modo SEGURO)       resultado + imagen
                               (modo DUDA MEDIA /
                                NO CONCLUYENTE)
                                    │
                                    ▼
                          Explicación en lenguaje
                          natural con nivel de
                          confianza explicado
```

En el módulo de mascotas la cascada tiene un paso adicional: primero se clasifica especie (perro/gato) y solo después se clasifica raza con un modelo específico para esa especie.

## Módulo 1: Clasificador de mascotas

**Pipeline jerárquico de 3 modelos:**

1. **Fase 1 — Especie** (perro vs. gato): CNN secuencial entrenada desde cero, imágenes 224×224. Se probaron 3 variantes de profundidad; se seleccionó la de 5 capas convolucionales por mejor desempeño (ver [Resultados](#resultados)).
2. **Fase 2a — Raza de perro** (25 clases): transfer learning con MobileNetV2 + fine-tuning.
3. **Fase 2b — Raza de gato** (12 clases): transfer learning con MobileNetV2 + fine-tuning.

Dataset: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Módulo 2: Clasificador OCT retinal

CNN de 5 bloques convolucionales (Conv2D + BatchNormalization + MaxPool2D en cada bloque, GlobalAveragePooling2D + Dense con Dropout en la cabeza) entrenada desde cero para clasificar imágenes OCT en escala de grises (160×160) en 4 clases clínicas: **CNV, DME, DRUSEN, NORMAL**.

Este módulo es de carácter educativo/demostrativo — no sustituye valoración oftalmológica profesional.

## Capa de razonamiento con LLM

Ambos módulos usan GPT-4o-mini para generar una explicación en lenguaje natural del resultado, pero con reglas de activación distintas:

- **Mascotas:** el LLM solo recibe la imagen (visión) cuando la confianza del clasificador es baja. Con confianza alta, el LLM redacta la explicación basándose únicamente en los datos numéricos del modelo — no repite la inferencia visual innecesariamente. Umbrales:

  | Modo | Confianza especie | Confianza raza |
  |---|---|---|
  | SEGURO | ≥ 0.80 | ≥ 0.75 |
  | DUDA MEDIA | ≥ 0.65 | ≥ 0.55 |
  | NO CONCLUYENTE | < 0.65 | < 0.55 |

- **OCT:** el LLM siempre genera una explicación educativa de 8 secciones (resultado, causas, síntomas, tratamientos generales, limitaciones, etc.), con reglas explícitas anti-alucinación en el prompt (no da diagnóstico definitivo, no recomienda medicamentos ni dosis, aclara que no sustituye valoración médica).

Esta decisión de diseño —limitar cuándo el LLM ve la imagen— reduce costo y latencia, y evita que el modelo de lenguaje contradiga innecesariamente a una predicción ya confiable.

## Resultados

### Fase 1 — Especie (perro/gato)

| Variante | Test accuracy |
|---|---|
| 3 capas | 74.9% |
| 4 capas | 83.6% |
| **5 capas (usada en producción)** | **88.1%** |

### Fase 2 — Raza

| Modelo | Transfer learning | Fine-tuning |
|---|---|---|
| Perros (25 razas) | 93% | 93% |
| Gatos (12 razas) | 87% | 90% |

### OCT retinal (4 clases)

| Métrica | Valor |
|---|---|
| Test accuracy | 93.9% |
| F1 macro avg | 0.92 |

Detalle de precision/recall por clase en [`docs/reporte.md`](docs/reporte.md).

## Estructura del repositorio

```
├── app/                  → código de la app Streamlit (Hugging Face Space)
├── models/                → pesos entrenados (.h5) + README con detalle de cada modelo
├── notebooks/             → notebooks de entrenamiento, uno por modelo
├── docs/                  → reporte técnico y diagrama de arquitectura
└── assets/                → capturas de la demo
```

## Cómo correr localmente

```bash
git clone <url-del-repo>
cd Caso-Negocio-CNN
pip install -r requirements.txt

export OPENAI_API_KEY=tu_api_key
export DATA_DIR=./data       # solo necesario para re-entrenar
export MODELS_DIR=./models

streamlit run app/app.py
```

## Limitaciones y trabajo futuro

- **Interpretabilidad visual (Grad-CAM):** no implementado en esta entrega por restricción de tiempo. Permitiría visualizar qué regiones de la imagen influyeron en cada predicción, más allá del score numérico.
- **Evaluación cuantitativa de las explicaciones GPT:** actualmente no hay una métrica formal de calidad/fidelidad de las explicaciones generadas por el LLM respecto a la predicción del modelo.
- **Segmentación:** el dataset Oxford-IIIT Pet incluye trimaps de segmentación no utilizados en esta versión; podría extenderse el módulo de mascotas a segmentación además de clasificación.
- **Desbalance de clases:** algunas razas y la clase DRUSEN en OCT presentan menor precision/recall relativo — candidatas a técnicas de balanceo adicionales (oversampling, focal loss).
