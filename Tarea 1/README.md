# Análisis estadístico comparativo de corpus literario: Jane Austen vs. G. K. Chesterton

Tarea de estadística sobre origen de datos textual. Se comparan seis obras de dominio
público (tres novelas de Jane Austen y tres obras de G. K. Chesterton) usando
estadística descriptiva, frecuencias de palabras, n-gramas y uso de puntuación.

## Estructura de la carpeta

```
├── README.md                 <- este archivo
├── analisis_textual.ipynb    <- notebook con el pipeline completo (código fuente)
├── austen-emma.txt           <- corpus fuente (.txt, dominio público)
├── austen-persuasion.txt
├── austen-sense.txt
├── chesterton-ball.txt
├── chesterton-brown.txt
├── chesterton-thursday.txt
├── results/                   <- salidas generadas al correr el notebook
│   ├── estadisticas_por_obra.csv
│   ├── estadisticas_por_autor.csv
│   ├── top_palabras_por_autor.csv
│   ├── ngramas_por_autor.csv
│   ├── puntuacion_por_autor.csv
│   ├── pruebas_estadisticas.json
│   └── fig_*.png
└── report/
    ├── report.tex              <- reporte científico (LaTeX)
    └── report.pdf              <- reporte compilado
```

## Fuente de datos

Corpus `gutenberg` incluido en NLTK (subconjunto curado de Project Gutenberg,
dominio público): 3 novelas de Jane Austen y 3 obras de G. K. Chesterton.
Referencia: Bird, Klein & Loper (2009), *Natural Language Processing with Python*.

## Cómo reproducir

Abrir `analisis_textual.ipynb` y correr todas las celdas. La primera celda
descarga automáticamente los recursos de NLTK necesarios (`punkt_tab`,
`stopwords`) e instala/usa: `nltk`, `pandas`, `matplotlib`, `scipy`.

El notebook asume que los 6 `.txt` están en la misma carpeta que él, y crea
`results/` automáticamente ahí mismo al ejecutarse.

## Compilar el reporte

```bash
cd report
pdflatex report.tex && pdflatex report.tex
```

## Resumen de hallazgos

- **Longitud de oración**: Austen usa oraciones más largas y variables
  (media 22.9 palabras, desv. est. 21.4) que Chesterton (media 17.9, desv. est. 12.4).
  Diferencia significativa (Mann–Whitney U, p = 2.7×10⁻²⁷).
- **Riqueza léxica**: usando STTR (comparable entre obras de distinta longitud),
  Chesterton tiene vocabulario ligeramente más diverso (~0.43) que Austen (~0.40).
- **Puntuación**: Austen usa más punto y coma y guión largo (periodos complejos
  encadenados); Chesterton usa ligeramente más signos de interrogación.
  Diferencia significativa (χ², p ≪ 0.001).
- **N-gramas**: en Austen dominan fórmulas de tratamiento social (*mr. knightley*,
  *mrs. weston*); en Chesterton domina el patrón de diálogo (*said + nombre propio*).
- Los patrones son consistentes **dentro** de las 3 obras de cada autor, lo que
  sugiere que reflejan estilo de autor y no particularidades de una sola obra.

Detalle completo, metodología y limitaciones en `report/report.pdf`.
