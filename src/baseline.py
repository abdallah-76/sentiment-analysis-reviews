"""Modèle de référence : TF-IDF + régression logistique.

Rapide à entraîner (quelques minutes sur CPU), facile à expliquer, et surtout
interprétable : on peut lister les mots qui poussent vers chaque classe.

Usage :
    python src/baseline.py
    python src/baseline.py --train-size 20000      # pour comparer à DistilBERT à données égales
    python src/baseline.py --csv mes_avis.csv      # si Hugging Face n'est pas accessible
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from data import LABELS, ROOT, load_splits
from metrics import RESULTS_DIR, evaluate


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),  # mots seuls + paires de mots ("not good")
                    min_df=2,
                    max_features=200_000,
                    sublinear_tf=True,
                ),
            ),
            # class_weight="balanced" : la classe "neutre" est 2x plus rare que les autres
            ("clf", LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")),
        ]
    )


def top_words(pipe: Pipeline, k: int = 10) -> str:
    """Tableau markdown des mots les plus associés à chaque classe."""
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coefs = pipe.named_steps["clf"].coef_
    lines = ["| Classe | Mots les plus associés |", "|---|---|"]
    for i, label in enumerate(LABELS):
        best = np.argsort(coefs[i])[::-1][:k]
        lines.append(f"| {label} | {', '.join(names[best])} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--csv", help="CSV local (colonnes text, label) à la place du dataset Hugging Face")
    parser.add_argument("--train-size", type=int, default=None, help="Nombre d'avis d'entraînement (défaut : tous)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, val, test = load_splits(args.csv, args.train_size, args.seed)
    print(f"Train : {len(train)} | Validation : {len(val)} | Test : {len(test)}")

    pipe = build_pipeline()
    start = time.time()
    pipe.fit(train["text"], train["label"])
    train_seconds = time.time() - start
    print(f"Entraînement terminé en {train_seconds:.0f} s")

    val_f1 = f1_score(val["label"], pipe.predict(val["text"]), average="macro")
    print(f"Macro-F1 sur validation : {val_f1:.3f}")

    evaluate("baseline", test["text"], test["label"], pipe.predict(test["text"]),
             train_seconds, n_train=len(train))

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "top_words_baseline.md").write_text(top_words(pipe) + "\n", encoding="utf-8")
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / "baseline.joblib")
    print("Modèle sauvegardé dans models/baseline.joblib")


if __name__ == "__main__":
    main()
