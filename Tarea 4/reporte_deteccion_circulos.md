# Detección de centros de objetos circulares mediante Transformada de Hough

**Autor:** Ferre
**Técnica:** `cv2.HoughCircles` (Hough Circle Transform), replicando y extendiendo el ejemplo de monedas visto en la Clase 4.

## 1. Objetivo

Encontrar automáticamente el centro (x, y) y el radio de objetos circulares en imágenes, usando el mismo pipeline visto en clase, aplicado a tres imágenes de dominios distintos para evaluar qué tan bien generaliza la técnica y qué parámetros hay que ajustar en cada caso.

## 2. Metodología (pipeline)

Para cada imagen se aplicó la misma secuencia:

1. **Escala de grises** — `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
2. **Desenfoque Gaussiano** — `cv2.GaussianBlur` para reducir ruido de alta frecuencia que genera falsos bordes.
3. **Detección de círculos** — `cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius)`, donde:
   - `dp`: resolución del acumulador respecto a la imagen.
   - `minDist`: distancia mínima entre centros detectados (evita duplicados sobre el mismo objeto).
   - `param1`: umbral alto del detector de bordes interno (Canny).
   - `param2`: umbral del acumulador — qué tan "seguro" debe estar el algoritmo de que un conjunto de bordes forma un círculo. Es el parámetro más sensible: bajarlo detecta más círculos pero también más falsos positivos.
   - `minRadius` / `maxRadius`: rango de tamaño esperado del objeto, en píxeles.
4. **Marcado de resultados** — se dibuja el contorno detectado (verde) y el centro (rojo) sobre la imagen original.

El código completo y reproducible está en `deteccion_circulos.ipynb`.

> **Nota sobre las imágenes:** no conté con fotos propias de objetos circulares al momento de hacer la tarea, así que usé tres imágenes reales de dominio público (fuente: dataset estándar de `scikit-image`, originalmente provenientes de bancos de imágenes de Internet): monedas, un reloj y una superficie lunar. Esto permite mostrar la técnica en tres contextos distintos en vez de repetir el mismo ejemplo de clase.

## 3. Caso 1 — Monedas (análogo directo al ejemplo de clase)

**Parámetros:** `dp=1.2, minDist=40, param1=50, param2=30, minRadius=15, maxRadius=45`

![Pipeline monedas](imagenes/coins_proceso.png)

**Resultado:** 24 de 24 monedas detectadas correctamente (100%), incluyendo monedas de distinto tamaño y con relieve visual complejo (bordes irregulares por el grabado). El desenfoque gaussiano fue clave para que Canny no confundiera el detalle interno del grabado con el borde exterior de la moneda.

![Monedas detectadas](imagenes/coins_detectado.png)

Radios detectados: van de 18 a 31 px, consistente con las monedas más chicas y más grandes de la imagen.

## 4. Caso 2 — Reloj de pared (foto con *motion blur*)

**Parámetros:** `dp=1.2, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=200`

![Pipeline reloj](imagenes/clock_proceso.png)

**Resultado:** 1 círculo detectado, centro en `(208, 148)`, radio 52 px — coincide bien con el contorno visible del reloj a pesar del desenfoque de movimiento (motion blur) de la foto original. Esto muestra que la técnica es razonablemente robusta a desenfoque, siempre que el borde del objeto conserve algo de gradiente de intensidad.

![Reloj detectado](imagenes/clock_detectado.png)

## 5. Caso 3 — Cráteres lunares (dominio distinto: fotografía astronómica)

**Parámetros:** `dp=1.2, minDist=30, param1=60, param2=28, minRadius=8, maxRadius=40, blur=5`

![Pipeline luna](imagenes/moon_proceso.png)

**Resultado:** 3 cráteres detectados. El cráter grande superior izquierdo (centro `(119, 87)`, radio 25 px) se detecta con alta confianza porque tiene un borde circular limpio e iluminación uniforme. Los dos cráteres inferiores (radios 12–15 px) están en una zona de sombras irregulares y se detectaron con más ruido — es el caso donde más tuve que ajustar `param2` y `minDist` hacia arriba para eliminar falsos positivos (formas irregulares que casi parecían círculos).

![Luna detectada](imagenes/moon_detectado.png)

## 6. Hallazgos y conclusiones

- **La calidad de la detección depende fuertemente de `param2` y del preprocesamiento (blur).** Con texturas ruidosas (superficie lunar, grabado de las monedas) sin desenfoque suficiente, el algoritmo genera falsos positivos porque interpreta detalle interno como borde de otro círculo más chico.
- **No hay un único set de parámetros universal.** Los tres casos necesitaron rangos de radio y umbrales distintos porque la escala de los objetos (en píxeles) y el contraste con el fondo cambian mucho entre dominios (objeto fotografiado de cerca vs. superficie planetaria vs. reloj con blur).
- **`minDist` es crítico cuando los objetos están muy juntos** (caso monedas, caso cráteres agrupados): si es muy chico, un mismo círculo se detecta varias veces con leves variaciones de centro/radio.
- **La técnica es robusta a cierto grado de desenfoque** (caso reloj), lo cual tiene sentido porque Hough Circles trabaja sobre gradientes de borde, no sobre nitidez absoluta.
- **Limitación observada:** en zonas con sombras irregulares (cráteres inferiores del caso 3), el algoritmo puede confundir una forma no perfectamente circular con un círculo si el umbral (`param2`) es demasiado permisivo — hay que revisar visualmente el resultado, no solo confiar en el conteo.

## 7. Archivos entregados

- `reporte_deteccion_circulos.md` — este reporte.
- `deteccion_circulos.py` — script reproducible (descarga las 3 imágenes de muestra, corre el pipeline y genera todas las figuras).
- `imagenes/` — imágenes de proceso y resultado por caso.
