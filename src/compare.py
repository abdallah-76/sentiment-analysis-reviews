"""Compare les deux modèles : tableau markdown + graphique.

Usage (après avoir lancé baseline.py et transformer.py) :
    python src/compare.py

Le tableau généré (results/comparison.md) peut être collé tel quel dans le README.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import LABELS
from metrics import RESULTS_DIR

NICE_NAMES = {"baseline": "TF-IDF + Régression logistique", "distilbert": "DistilBERT (fine-tuné)"}


def load_all() -> list[dict]:
    found = []
    for name in ("baseline", "distilbert"):
        path = RESULTS_DIR / f"{name}_metrics.json"
        if path.exists():
            found.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            print(f"(absent : {path.name} – lance d'abord le script correspondant)")
    return found


def format_duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def to_markdown(results: list[dict]) -> str:
    header = ["Modèle", "Avis d'entraînement", "Accuracy", "F1 macro"] + [f"F1 {l}" for l in LABELS] + ["Temps d'entraînement"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        row = [
            NICE_NAMES.get(r["model"], r["model"]),
            f"{r['n_train']:,}".replace(",", " "),
            f"{r['accuracy']:.1%}",
            f"{r['macro_f1']:.3f}",
            *[f"{r['f1_per_class'][l]:.3f}" for l in LABELS],
            format_duration(r["train_seconds"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    for k, (key, title) in enumerate([("accuracy", "Accuracy"), ("macro_f1", "F1 macro")]):
        values = [r[key] for r in results]
        positions = [i + (k - 0.5) * width for i in range(len(results))]
        bars = ax.bar(positions, values, width, label=title)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(results)), [NICE_NAMES.get(r["model"], r["model"]) for r in results])
    ax.set_ylim(0, 1.18)  # marge pour les valeurs affichées et la légende
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Comparaison des modèles sur le jeu de test")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    results = load_all()
    if not results:
        raise SystemExit("Aucun résultat trouvé dans results/. Lance baseline.py puis transformer.py.")
    table = to_markdown(results)
    (RESULTS_DIR / "comparison.md").write_text(table + "\n", encoding="utf-8")
    plot(results)
    print(table)
    print("\nÉcrit : results/comparison.md et results/comparison.png")


if __name__ == "__main__":
    main()
