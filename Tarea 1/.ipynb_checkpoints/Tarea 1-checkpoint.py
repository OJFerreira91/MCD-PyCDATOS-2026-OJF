"""
Análisis estadístico comparativo de corpus literario
Autores: Jane Austen (3 novelas) vs G. K. Chesterton (3 obras)
Fuente: Gutenberg Corpus (subset incluido en NLTK), textos de dominio público.

Genera todas las tablas (CSV) y figuras (PNG) usadas en el reporte.
"""
import os
import re
import json
import string
from collections import Counter, defaultdict

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "nltk_data")
nltk.data.path.append(os.path.abspath(NLTK_DATA_PATH))

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

STOPWORDS = set(stopwords.words("english"))

# ------------------------------------------------------------------
# Corpus: dos autores, tres obras cada uno
# ------------------------------------------------------------------
CORPUS = {
    "Austen": {
        "Emma (1816)": "austen-emma.txt",
        "Persuasion (1817)": "austen-persuasion.txt",
        "Sense and Sensibility (1811)": "austen-sense.txt",
    },
    "Chesterton": {
        "The Ball and the Cross (1909)": "chesterton-ball.txt",
        "The Innocence of Father Brown (1911)": "chesterton-brown.txt",
        "The Man Who Was Thursday (1908)": "chesterton-thursday.txt",
    },
}

PUNCT_MARKS = [",", ";", ":", "!", "?", "-", "--", "(", ")", '"', "'"]


def load_text(filename):
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            text = f.read()
    # Quitar la línea de título tipo "[Emma by Jane Austen 1816]" que
    # antepone el corpus de NLTK y no es parte de la prosa del autor.
    text = re.sub(r"^\s*\[.*?\]\s*", "", text, count=1, flags=re.DOTALL)
    return text.strip()


def basic_stats(text):
    sentences = sent_tokenize(text)
    tokens_raw = word_tokenize(text)
    words = [t.lower() for t in tokens_raw if re.search(r"[a-zA-Z]", t)]
    n_words = len(words)
    n_sentences = len(sentences)
    n_chars = sum(len(w) for w in words)
    vocab = set(words)
    n_vocab = len(vocab)
    freq = Counter(words)
    hapax = sum(1 for w, c in freq.items() if c == 1)

    sent_lengths = [
        len([t for t in word_tokenize(s) if re.search(r"[a-zA-Z]", t)])
        for s in sentences
    ]
    word_lengths = [len(w) for w in words]

    punct_counts = Counter(t for t in tokens_raw if t in PUNCT_MARKS)
    # Signos por 1000 palabras, para comparar longitudes distintas de obra
    punct_per_1000 = {p: (punct_counts.get(p, 0) / n_words) * 1000 for p in PUNCT_MARKS}
    punct_raw = {p: punct_counts.get(p, 0) for p in PUNCT_MARKS}

    return {
        "n_words": n_words,
        "n_sentences": n_sentences,
        "avg_sentence_len": np.mean(sent_lengths),
        "std_sentence_len": np.std(sent_lengths),
        "avg_word_len": np.mean(word_lengths),
        "vocab_size": n_vocab,
        "ttr": n_vocab / n_words,  # type-token ratio
        "hapax_ratio": hapax / n_vocab,
        "punct_per_1000_words": punct_per_1000,
        "punct_raw": punct_raw,
        "sent_lengths": sent_lengths,
        "word_freq": freq,
        "tokens_raw": tokens_raw,
        "words": words,
    }


def standardized_ttr(words, window=1000):
    """TTR estandarizado (STTR): promedio del TTR calculado en bloques de
    tamaño fijo. Evita el sesgo del TTR clásico, que cae artificialmente
    en textos largos (ley de Heaps) y por tanto no es comparable entre
    obras/corpus de longitudes distintas."""
    chunks = [words[i:i + window] for i in range(0, len(words), window)]
    chunks = [c for c in chunks if len(c) == window]  # descartar el último bloque incompleto
    if not chunks:
        return np.nan
    ratios = [len(set(c)) / len(c) for c in chunks]
    return float(np.mean(ratios))


def top_ngrams(words, n=2, top_k=15, filter_stopwords=True):
    if filter_stopwords:
        seq = [w for w in words if w not in STOPWORDS and w not in string.punctuation]
    else:
        seq = words
    ngrams = zip(*[seq[i:] for i in range(n)])
    counts = Counter(ngrams)
    return counts.most_common(top_k)


def main():
    rows = []
    per_work_stats = {}
    author_word_pool = defaultdict(list)
    author_sentlen_pool = defaultdict(list)
    author_punct_raw = defaultdict(lambda: Counter())

    for author, works in CORPUS.items():
        for title, filename in works.items():
            text = load_text(filename)
            stats_d = basic_stats(text)
            per_work_stats[title] = stats_d
            author_word_pool[author].extend(stats_d["words"])
            author_sentlen_pool[author].extend(stats_d["sent_lengths"])
            author_punct_raw[author].update(stats_d["punct_raw"])

            rows.append({
                "author": author,
                "title": title,
                "n_words": stats_d["n_words"],
                "n_sentences": stats_d["n_sentences"],
                "avg_sentence_len": round(stats_d["avg_sentence_len"], 2),
                "std_sentence_len": round(stats_d["std_sentence_len"], 2),
                "avg_word_len": round(stats_d["avg_word_len"], 3),
                "vocab_size": stats_d["vocab_size"],
                "ttr": round(stats_d["ttr"], 4),
                "sttr_1000": round(standardized_ttr(stats_d["words"], 1000), 4),
                "hapax_ratio": round(stats_d["hapax_ratio"], 4),
                **{f"punct_{k}_per1000": round(v, 2) for k, v in stats_d["punct_per_1000_words"].items()},
            })
            print(f"[OK] {title}: {stats_d['n_words']} palabras, {stats_d['n_sentences']} oraciones")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "estadisticas_por_obra.csv"), index=False)

    # ---------------- Agregado por autor ----------------
    agg_rows = []
    for author in CORPUS:
        w = author_word_pool[author]
        freq = Counter(w)
        vocab = set(w)
        hapax = sum(1 for k, c in freq.items() if c == 1)
        sl = author_sentlen_pool[author]
        agg_rows.append({
            "author": author,
            "n_words_total": len(w),
            "vocab_size": len(vocab),
            "ttr": round(len(vocab) / len(w), 4),
            "sttr_1000": round(standardized_ttr(w, 1000), 4),
            "hapax_ratio": round(hapax / len(vocab), 4),
            "avg_sentence_len": round(np.mean(sl), 2),
            "std_sentence_len": round(np.std(sl), 2),
            "median_sentence_len": round(np.median(sl), 2),
        })
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(RESULTS_DIR, "estadisticas_por_autor.csv"), index=False)

    # ---------------- Top palabras (sin stopwords) por autor ----------------
    top_words_rows = []
    for author in CORPUS:
        w = [x for x in author_word_pool[author] if x not in STOPWORDS and x.isalpha()]
        freq = Counter(w)
        for word, count in freq.most_common(20):
            top_words_rows.append({"author": author, "word": word, "count": count,
                                    "per_1000": round(count / len(author_word_pool[author]) * 1000, 3)})
    pd.DataFrame(top_words_rows).to_csv(os.path.join(RESULTS_DIR, "top_palabras_por_autor.csv"), index=False)

    # ---------------- N-gramas (bigramas y trigramas) ----------------
    ngram_rows = []
    for author in CORPUS:
        w = author_word_pool[author]
        for n in (2, 3):
            for gram, count in top_ngrams(w, n=n, top_k=15):
                ngram_rows.append({
                    "author": author, "n": n,
                    "ngram": " ".join(gram), "count": count,
                })
    pd.DataFrame(ngram_rows).to_csv(os.path.join(RESULTS_DIR, "ngramas_por_autor.csv"), index=False)

    # ---------------- Puntuación agregada por autor (por 1000 palabras) ----
    punct_rows = []
    for author in CORPUS:
        works_titles = list(CORPUS[author].keys())
        sub = df[df.title.isin(works_titles)]
        for mark in PUNCT_MARKS:
            col = f"punct_{mark}_per1000"
            if col in sub.columns:
                punct_rows.append({
                    "author": author, "mark": mark,
                    "mean_per_1000_words": round(sub[col].mean(), 3),
                })
    df_punct = pd.DataFrame(punct_rows)
    df_punct.to_csv(os.path.join(RESULTS_DIR, "puntuacion_por_autor.csv"), index=False)

    # ---------------- Pruebas estadísticas ----------------
    tests = {}
    sl_austen = author_sentlen_pool["Austen"]
    sl_chesterton = author_sentlen_pool["Chesterton"]
    u_stat, p_mw = stats.mannwhitneyu(sl_austen, sl_chesterton, alternative="two-sided")
    t_stat, p_t = stats.ttest_ind(sl_austen, sl_chesterton, equal_var=False)
    tests["sentence_length_mannwhitney"] = {"U": float(u_stat), "p_value": float(p_mw)}
    tests["sentence_length_welch_t"] = {"t": float(t_stat), "p_value": float(p_t)}

    # Chi-cuadrada sobre distribución conjunta de signos de puntuación
    # (usa CONTEOS absolutos, no tasas, como exige la prueba)
    contingency = pd.DataFrame({
        author: [author_punct_raw[author].get(m, 0) for m in PUNCT_MARKS]
        for author in CORPUS
    }, index=PUNCT_MARKS).T
    contingency = contingency.loc[:, (contingency.sum(axis=0) > 0)]  # descartar signos ausentes en ambos autores
    chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
    tests["punctuation_chi2"] = {"chi2": float(chi2), "p_value": float(p_chi), "dof": int(dof)}

    with open(os.path.join(RESULTS_DIR, "pruebas_estadisticas.json"), "w") as f:
        json.dump(tests, f, indent=2, ensure_ascii=False)

    print("\n--- Pruebas estadísticas ---")
    print(json.dumps(tests, indent=2))

    # ================= FIGURAS =================
    plt.rcParams["figure.dpi"] = 140

    # 1. Boxplot de longitud de oración por autor
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([sl_austen, sl_chesterton], labels=["Austen", "Chesterton"], showfliers=False)
    ax.set_ylabel("Palabras por oración")
    ax.set_title("Distribución de longitud de oración por autor")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_boxplot_sentence_length.png"))
    plt.close(fig)

    # 2. Barras: TTR y hapax ratio por obra
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0" if a == "Austen" else "#DD8452" for a in df["author"]]
    ax.bar(df["title"], df["sttr_1000"], color=colors)
    ax.set_ylabel("STTR (ventanas de 1000 palabras)")
    ax.set_title("Riqueza léxica estandarizada por obra")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_ttr_por_obra.png"))
    plt.close(fig)

    # 3. Puntuación por autor (barras agrupadas)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = df_punct.pivot(index="mark", columns="author", values="mean_per_1000_words")
    pivot = pivot.reindex(PUNCT_MARKS)
    x = np.arange(len(pivot.index))
    width = 0.35
    ax.bar(x - width / 2, pivot["Austen"], width, label="Austen", color="#4C72B0")
    ax.bar(x + width / 2, pivot["Chesterton"], width, label="Chesterton", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Ocurrencias por 1000 palabras")
    ax.set_title("Uso de signos de puntuación por autor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_puntuacion_autor.png"))
    plt.close(fig)

    # 4. Top-15 palabras de contenido por autor (subplots)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, author, color in zip(axes, CORPUS.keys(), ["#4C72B0", "#DD8452"]):
        sub = pd.DataFrame(top_words_rows)
        sub = sub[sub.author == author].head(15).sort_values("count")
        ax.barh(sub["word"], sub["count"], color=color)
        ax.set_title(f"Top-15 palabras de contenido — {author}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_top_palabras.png"))
    plt.close(fig)

    print(f"\nResultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
