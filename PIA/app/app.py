import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from openai import OpenAI
from io import BytesIO
import base64
import os
import pandas as pd

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(
    page_title="AI Vision Suite",
    page_icon="🧠",
    layout="centered"
)

IMG_SIZE = (224, 224)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODELO_GPT = "gpt-4o-mini"

UMBRAL_ESPECIE_SEGURA = 0.80
UMBRAL_RAZA_SEGURA = 0.75
UMBRAL_ESPECIE_MEDIA = 0.65
UMBRAL_RAZA_MEDIA = 0.55

# ------------------------
# OPENAI CLIENT
# ------------------------
@st.cache_resource
def cargar_cliente_openai():
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        st.error("Falta OPENAI_API_KEY en Hugging Face Secrets.")
        st.stop()

    return OpenAI(api_key=api_key)

# ------------------------
# CARGA DE MODELOS MASCOTAS
# ------------------------
@st.cache_resource
def cargar_fase1():
    path = MODELS_DIR / "5capas_modelo_fase1_perro_gato_TF215.h5"
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def cargar_perros():
    path = MODELS_DIR / "modelo_razas_perros_FineTuning.h5"
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def cargar_gatos():
    path = MODELS_DIR / "modelo_razas_gatos_FineTuning.h5"
    return tf.keras.models.load_model(path, compile=False)

# ------------------------
# RAZAS
# ------------------------
razas_perros = [
    "American bulldog", "American pitbull", "Basset hound", "Beagle", "Boxer",
    "Chihuahua", "English cocker", "English setter", "German", "Great pyrenees",
    "Havanese", "Japanese chin", "Keeshond", "Leonberger", "Miniature pinscher",
    "Newfoundland", "Pomeranian", "Pug", "Saint bernard", "Samoyed",
    "Scottish terrier", "Shiba inu", "Staffordshire bullterrier",
    "Wheaten terrier", "Yorkshire terrier"
]

razas_gatos = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx"
]

# ------------------------
# FUNCIONES CNN MASCOTAS
# ------------------------
def preparar_imagen_fase1(imagen):
    imagen = imagen.convert("RGB")
    imagen = imagen.resize(IMG_SIZE)
    imagen_array = np.array(imagen, dtype=np.float32) / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)
    return imagen_array

def preparar_imagen_razas(imagen):
    imagen = imagen.convert("RGB")
    imagen = imagen.resize(IMG_SIZE)
    imagen_array = np.array(imagen, dtype=np.float32)
    imagen_array = np.expand_dims(imagen_array, axis=0)
    return imagen_array

def predecir_mascota(imagen_array):
    modelo_fase1 = cargar_fase1()
    pred = modelo_fase1.predict(imagen_array, verbose=0)
    pred = float(np.ravel(pred)[0])

    prob_gato = pred
    prob_perro = 1 - pred

    if prob_gato >= prob_perro:
        return "Gato", "🐱", prob_gato
    else:
        return "Perro", "🐶", prob_perro

def predecir_raza(tipo_mascota, imagen_array):
    if tipo_mascota == "Perro":
        modelo = cargar_perros()
        pred_raza = modelo.predict(imagen_array, verbose=0)[0]
        indice = int(np.argmax(pred_raza))
        return razas_perros[indice], float(pred_raza[indice])
    else:
        modelo = cargar_gatos()
        pred_raza = modelo.predict(imagen_array, verbose=0)[0]
        indice = int(np.argmax(pred_raza))
        return razas_gatos[indice], float(pred_raza[indice])

# ------------------------
# FUNCIONES GPT MASCOTAS
# ------------------------
def determinar_modo_gpt(confianza_mascota, confianza_raza):
    if confianza_mascota >= UMBRAL_ESPECIE_SEGURA and confianza_raza >= UMBRAL_RAZA_SEGURA:
        return "SEGURO"
    elif confianza_mascota >= UMBRAL_ESPECIE_MEDIA and confianza_raza >= UMBRAL_RAZA_MEDIA:
        return "DUDA MEDIA"
    else:
        return "NO CONCLUYENTE"

def debe_gpt_ver_imagen(confianza_mascota, confianza_raza):
    return confianza_mascota < UMBRAL_ESPECIE_SEGURA or confianza_raza < UMBRAL_RAZA_SEGURA

def imagen_a_base64_gpt(imagen, max_lado=768):
    img = imagen.convert("RGB")
    img.thumbnail((max_lado, max_lado))

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)

    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"

def construir_prompt_gpt(
    tipo_mascota,
    raza,
    confianza_mascota,
    confianza_raza,
    modo_gpt,
    gpt_ve_imagen
):
    return f"""
Actúa como un asistente experto en mascotas dentro de una aplicación de visión computacional.

El sistema usa una CNN de dos fases:
Fase 1: clasificación de especie.
Fase 2: clasificación de raza.

IMPORTANTE:
- No contradigas agresivamente al modelo.
- Si hay incertidumbre, habla con cautela.
- No des diagnósticos veterinarios.
- No inventes certeza absoluta.
- Responde en español.
- Usa exactamente la estructura solicitada.
- Si GPT no ve imagen, basa tu respuesta únicamente en los datos del modelo.
- Si GPT ve imagen, úsala solo como apoyo visual prudente, no como reemplazo absoluto del modelo.

Datos del sistema:
Especie detectada: {tipo_mascota}
Raza estimada: {raza}
Confianza especie: {confianza_mascota*100:.2f}%
Confianza raza: {confianza_raza*100:.2f}%
Modo de respuesta: {modo_gpt}
GPT ve imagen: {"Sí" if gpt_ve_imagen else "No"}

Reglas de redacción según modo:

1. Si el modo es SEGURO:
- Presenta el resultado como detectado.
- Usa tono positivo y profesional.
- Indica que la predicción tiene alta confianza.

2. Si el modo es DUDA MEDIA:
- Presenta el resultado como probable.
- Menciona que puede haber similitud con razas parecidas.
- No afirmes que sea definitivo.

3. Si el modo es NO CONCLUYENTE:
- Indica que el resultado no es concluyente.
- No te enfoques demasiado en la raza exacta.
- Recomienda subir una imagen más clara, frontal y bien iluminada.

Estructura obligatoria:

🐾 Resultado:
Una línea breve con especie, raza y estado del resultado.

Características principales:
1. ...
2. ...
3. ...
4. ...
5. ...
6. ...

Cuidados y consideraciones:
1. ...
2. ...
3. ...
4. ...
5. ...
6. ...

Dato interesante:
Un dato breve sobre la raza o, si no es concluyente, sobre mascotas con rasgos mixtos.

Nivel de confianza:
Explica el nivel de confianza según el modo.

Recomendación final:
Una recomendación breve y útil para el usuario.

Extensión máxima: 260 palabras.
"""

def generar_respuesta_gpt(tipo_mascota, raza, confianza_mascota, confianza_raza, imagen):
    client = cargar_cliente_openai()

    modo_gpt = determinar_modo_gpt(confianza_mascota, confianza_raza)
    gpt_ve_imagen = debe_gpt_ver_imagen(confianza_mascota, confianza_raza)

    prompt = construir_prompt_gpt(
        tipo_mascota=tipo_mascota,
        raza=raza,
        confianza_mascota=confianza_mascota,
        confianza_raza=confianza_raza,
        modo_gpt=modo_gpt,
        gpt_ve_imagen=gpt_ve_imagen
    )

    if gpt_ve_imagen:
        imagen_data_url = imagen_a_base64_gpt(imagen)

        contenido_usuario = [
            {"type": "input_text", "text": prompt},
            {
                "type": "input_image",
                "image_url": imagen_data_url,
                "detail": "low"
            }
        ]
    else:
        contenido_usuario = [
            {"type": "input_text", "text": prompt}
        ]

    response = client.responses.create(
        model=MODELO_GPT,
        input=[
            {
                "role": "user",
                "content": contenido_usuario
            }
        ],
        temperature=0.4,
        max_output_tokens=600
    )

    return modo_gpt, gpt_ve_imagen, response.output_text

# ------------------------
# INTERFAZ
# ------------------------
tab1, tab2 = st.tabs(["🐶🐱 Mascotas", "🧠 OCT"])

# ==========================================================
# TAB 1 - MASCOTAS
# ==========================================================
with tab1:
    st.title("🐾 Clasificador Inteligente de Mascotas")

    st.write(
        "Sube una foto de tu mascota y la inteligencia artificial identificará "
        "si es un gato o un perro, además de estimar su raza más probable."
    )

    st.markdown("---")

    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"],
        key="mascotas_uploader"
    )

    if archivo is not None:
        imagen = Image.open(archivo)
        st.image(imagen, caption="Imagen cargada", use_container_width=True)

        if st.button("Analizar imagen", key="btn_mascotas"):
            with st.spinner("Analizando imagen..."):
                imagen_fase1 = preparar_imagen_fase1(imagen)
                imagen_razas = preparar_imagen_razas(imagen)

                tipo_mascota, emoji, confianza_mascota = predecir_mascota(imagen_fase1)
                raza, confianza_raza = predecir_raza(tipo_mascota, imagen_razas)

                modo_gpt, gpt_ve_imagen, respuesta_gpt = generar_respuesta_gpt(
                    tipo_mascota=tipo_mascota,
                    raza=raza,
                    confianza_mascota=confianza_mascota,
                    confianza_raza=confianza_raza,
                    imagen=imagen
                )

                st.session_state["resultado_mascotas"] = {
                    "tipo_mascota": tipo_mascota,
                    "emoji": emoji,
                    "confianza_mascota": confianza_mascota,
                    "raza": raza,
                    "confianza_raza": confianza_raza,
                    "modo_gpt": modo_gpt,
                    "gpt_ve_imagen": gpt_ve_imagen,
                    "respuesta_gpt": respuesta_gpt
                }

        if "resultado_mascotas" in st.session_state:
            r = st.session_state["resultado_mascotas"]

            st.success("Análisis completado")
            st.markdown("---")
            st.subheader("Resultado")

            st.markdown(f"## {r['emoji']} Es un {r['tipo_mascota'].lower()}")
            st.markdown(f"### Raza probable: **{r['raza']}**")
            st.write(f"Confianza mascota: **{r['confianza_mascota'] * 100:.2f}%**")
            st.write(f"Confianza raza: **{r['confianza_raza'] * 100:.2f}%**")

            st.info(
                "Este resultado es una estimación basada en la imagen cargada. "
                "Puede variar según la iluminación, posición y calidad de la foto."
            )

            st.markdown("---")
            st.subheader("🧠 Análisis Generativo GPT")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Modo IA", r["modo_gpt"])
            with col2:
                st.metric("Imagen GPT", "Activado" if r["gpt_ve_imagen"] else "No requerido")

            if r["modo_gpt"] == "SEGURO":
                st.success(r["respuesta_gpt"])
            elif r["modo_gpt"] == "DUDA MEDIA":
                st.warning(r["respuesta_gpt"])
            else:
                st.error(r["respuesta_gpt"])

# ==========================================================
# TAB 2 - OCT
# ==========================================================
with tab2:
    st.title("🧠 RetinaScan AI")

    st.write(
        "Sistema de apoyo educativo basado en Deep Learning para clasificación "
        "de imágenes OCT retinales en CNV, DME, DRUSEN y NORMAL."
    )

    st.caption("Clasificación OCT retinal con Deep Learning + GPT")

    st.markdown("---")

    # ==========================================================
    # CONFIGURACIÓN OCT
    # ==========================================================
    RUTA_PESOS_OCT = MODELS_DIR / "pesos_oct.weights.h5"

    IMG_HEIGHT_OCT = 160
    IMG_WIDTH_OCT = 160
    CLASES_OCT = ["CNV", "DME", "DRUSEN", "NORMAL"]

    # ==========================================================
    # CARGA MODELO OCT CON PESOS
    # ==========================================================
    @st.cache_resource
    def cargar_modelo_oct():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(160, 160, 1)),
            tf.keras.layers.Rescaling(1./255),

            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.50),

            tf.keras.layers.Dense(4, activation="softmax")
        ])

        if not RUTA_PESOS_OCT.exists():
            st.error(f"No se encontró el archivo de pesos OCT: {RUTA_PESOS_OCT}")
            st.stop()

        model.load_weights(str(RUTA_PESOS_OCT))
        return model

    modelo_oct = cargar_modelo_oct()

    # ==========================================================
    # FUNCIONES OCT
    # ==========================================================
    def preparar_imagen_oct(uploaded_file):
        img = Image.open(uploaded_file).convert("L")
        img = img.resize((IMG_WIDTH_OCT, IMG_HEIGHT_OCT))

        # No dividir entre 255 porque el modelo ya tiene Rescaling(1./255)
        arr = np.array(img).astype("float32")
        arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)

        return img, arr

    def predecir_oct(arr):
        salida = modelo_oct.predict(arr, verbose=0)

        if len(salida.shape) != 2 or salida.shape[1] != len(CLASES_OCT):
            st.error(
                f"El modelo OCT no devolvió 4 probabilidades. "
                f"Output recibido: {salida.shape}"
            )
            st.stop()

        probs = salida[0]

        idx = int(np.argmax(probs))
        clase = CLASES_OCT[idx]
        confianza = float(probs[idx])

        orden = np.argsort(probs)[::-1]
        top1 = float(probs[orden[0]])
        top2 = float(probs[orden[1]])
        gap = top1 - top2

        return clase, confianza, probs, gap

    def nivel_confianza(confianza, gap):
        if confianza >= 0.88 and gap >= 0.10:
            return "🟢 Alta confianza"
        elif confianza >= 0.70:
            return "🟡 Confianza media"
        else:
            return "🔴 Resultado incierto"

    def color_resultado(clase):
        colores = {
            "CNV": "#2E86DE",
            "DME": "#8E44AD",
            "DRUSEN": "#D4AC0D",
            "NORMAL": "#27AE60"
        }
        return colores.get(clase, "#34495E")

    def determinar_modo_gpt_oct(confianza, gap):
        if confianza >= 0.88 and gap >= 0.10:
            return "SEGURO"
        elif confianza >= 0.70:
            return "DUDA MEDIA"
        else:
            return "NO CONCLUYENTE"

    def generar_explicacion_gpt_oct(clase, confianza, gap, probs):
        modo_gpt = determinar_modo_gpt_oct(confianza, gap)

        probs_texto = "\n".join([
            f"- {clase_i}: {prob*100:.2f}%"
            for clase_i, prob in zip(CLASES_OCT, probs)
        ])

        prompt = f"""
Actúa como asistente académico especializado en visión computacional aplicada a imágenes OCT de retina.

El sistema usa una red neuronal convolucional 5CNN para clasificar imágenes OCT en cuatro clases:
CNV, DME, DRUSEN y NORMAL.

Resultado del modelo:
Clase predicha: {clase}
Confianza: {confianza*100:.2f}%
Separación entre Top 1 y Top 2: {gap*100:.2f}%
Modo de respuesta: {modo_gpt}

Probabilidades:
{probs_texto}

Genera una explicación profesional en español con exactamente estas 8 secciones:

1. Resultado del modelo
2. Explicación breve de la categoría
3. Posibles causas o factores asociados
4. Qué suele originarla
5. Posibles síntomas asociados
6. Posibles tratamientos generales
7. Nivel de confianza del modelo
8. Limitaciones y recomendación final

Reglas:
- No des diagnóstico médico definitivo.
- No afirmes que el paciente tiene una enfermedad.
- Habla como predicción del modelo.
- No recomiendes medicamentos específicos ni dosis.
- Aclara que no sustituye valoración de un oftalmólogo.
- Máximo 280 palabras.
"""

        try:
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                return (
                    "No se encontró OPENAI_API_KEY. "
                    "Configura la variable de entorno en Hugging Face Secrets "
                    "para activar la explicación GPT dinámica."
                )

            client = OpenAI(api_key=api_key)

            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt}
                        ]
                    }
                ],
                temperature=0.35,
                max_output_tokens=750
            )

            return response.output_text

        except Exception as e:
            return f"No fue posible generar la explicación GPT. Error: {str(e)}"

    # ==========================================================
    # UPLOADER OCT
    # ==========================================================
    archivo_oct = st.file_uploader(
        "📤 Sube una imagen OCT",
        type=["jpg", "jpeg", "png"],
        key="oct_uploader"
    )

    if archivo_oct is not None:
        img_pil, arr = preparar_imagen_oct(archivo_oct)

        with st.spinner("Analizando imagen OCT..."):
            clase, confianza, probs, gap = predecir_oct(arr)

        estado = nivel_confianza(confianza, gap)
        color = color_resultado(clase)

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(
                img_pil,
                caption="Imagen OCT cargada",
                use_container_width=True
            )

        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color:#F8F9FA;
                    padding:25px;
                    border-radius:15px;
                    border-left:8px solid {color};
                ">
                    <h2 style="margin-bottom:10px;">Resultado estimado</h2>
                    <h1 style="color:{color}; margin-top:0px;">{clase}</h1>
                    <h3>Confianza: {confianza*100:.2f}%</h3>
                    <p style="font-size:18px;">{estado}</p>
                    <p>Separación Top 1 - Top 2: {gap*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### 📊 Probabilidades por clase")

        df_probs = pd.DataFrame({
            "Clase": CLASES_OCT,
            "Probabilidad": probs
        }).sort_values("Probabilidad", ascending=False)

        st.bar_chart(df_probs.set_index("Clase"))

        st.markdown("### 🧠 Interpretación educativa generada por GPT")

        with st.spinner("Generando explicación GPT..."):
            respuesta_gpt_oct = generar_explicacion_gpt_oct(
                clase=clase,
                confianza=confianza,
                gap=gap,
                probs=probs
            )

        respuesta_gpt_oct_html = respuesta_gpt_oct.replace("\n", "<br>")

        st.markdown(
            f"""
            <div style="
                background-color:#F8FAFC;
                padding:18px;
                border-radius:14px;
                border-left:6px solid {color};
                line-height:1.55;
                font-size:15px;
            ">
                {respuesta_gpt_oct_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("⚙️ Detalles técnicos del modelo"):
            st.write("• Arquitectura: CNN de 5 bloques")
            st.write("• Entrada: Imagen OCT 160x160 en escala de grises")
            st.write("• Salida: 4 clases: CNV, DME, DRUSEN y NORMAL")
            st.write("• Modelo cargado mediante arquitectura reconstruida + pesos `.weights.h5`")
            st.write("• Accuracy test aproximado: 93.9%")
            st.write("• La explicación GPT se genera dinámicamente con base en la predicción del modelo.")

        st.warning(
            "⚠️ Esta herramienta es educativa y demostrativa. "
            "No sustituye diagnóstico médico profesional ni valoración por oftalmología."
        )

    else:
        st.info("Carga una imagen OCT para comenzar.")