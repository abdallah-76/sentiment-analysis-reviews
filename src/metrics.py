"""Évaluation commune aux deux modèles : métriques, matrice de confusion, erreurs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import LABELS, ROOT

RESULTS_DIR = ROOT / "results"


def evaluate(name: str, texts, y_true, y_pred, train_seconds: float, n_train: int) -> dict:
    """Calcule et sauvegarde tout ce qu'il faut pour comparer les modèles."""
    RESULTS_DIR.mkdir(exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_class": {label: float(v) for label, v in zip(LABELS, per_class)},
        "train_seconds": round(float(train_seconds), 1),
        "n_train": int(n_train),
        "n_test": int(len(y_true)),
    }
    (RESULTS_DIR / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))

    _plot_confusion(name, y_true, y_pred)
    _save_errors(name, texts, y_true, y_pred)
    return metrics


def _plot_confusion(name: str, y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion – {name}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)


def _save_errors(name: str, texts, y_true, y_pred, k: int = 50) -> None:
    """Garde quelques avis mal classés pour l'analyse d'erreurs."""
    df = pd.DataFrame({"text": list(texts), "réel": y_true, "prédit": y_pred})
    errors = df[df["réel"] != df["prédit"]].copy()
    errors["réel"] = errors["réel"].map(dict(enumerate(LABELS)))
    errors["prédit"] = errors["prédit"].map(dict(enumerate(LABELS)))
    errors.head(k).to_csv(RESULTS_DIR / f"errors_{name}.csv", index=False, encoding="utf-8")
