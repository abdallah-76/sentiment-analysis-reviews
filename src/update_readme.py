"""Remplit la section « Résultats » du README avec les VRAIS résultats de tes runs.

À lancer après baseline.py, transformer.py (et idéalement avoir mis une capture
de la démo dans docs/demo.png) :
    python src/update_readme.py

Tout ce qui est écrit vient des fichiers de results/ : rien n'est inventé.
Le contenu entre <!-- RESULTS:START --> et <!-- RESULTS:END --> est remplacé.
"""
from __future__ import annotations

import re

import pandas as pd

from compare import NICE_NAMES, format_duration, load_all, plot, to_markdown
from data import LABELS, ROOT
from metrics import RESULTS_DIR

README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def observations(results: list[dict]) -> str:
    """Phrases factuelles déduites des chiffres (à compléter avec ton analyse)."""
    lines = []
    best = max(results, key=lambda r: r["macro_f1"])
    lines.append(
        f"- Meilleur F1 macro : **{NICE_NAMES.get(best['model'], best['model'])}** "
        f"({best['macro_f1']:.3f}, accuracy {best['accuracy']:.1%})."
    )
    if len(results) == 2:
        a, b = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
        gap = a["macro_f1"] - b["macro_f1"]
        lines.append(
            f"- Écart de F1 macro entre les deux modèles : **{gap:.3f}** "
            f"({a['n_train']:,} vs {b['n_train']:,} avis d'entraînement).".replace(",", " ")
        )
        slow, fast = sorted(results, key=lambda r: r["train_seconds"], reverse=True)
        ratio = f" (environ x{slow['train_seconds'] / fast['train_seconds']:.0f})" if fast["train_seconds"] >= 1 else ""
        lines.append(
            f"- Temps d'entraînement : {NICE_NAMES.get(slow['model'], slow['model'])} "
            f"{format_duration(slow['train_seconds'])} contre {format_duration(fast['train_seconds'])} "
            f"pour {NICE_NAMES.get(fast['model'], fast['model'])}{ratio}."
        )
    for r in results:
        worst = min(r["f1_per_class"], key=r["f1_per_class"].get)
        lines.append(
            f"- {NICE_NAMES.get(r['model'], r['model'])} : classe la plus difficile = "
            f"**{worst}** (F1 {r['f1_per_class'][worst]:.3f})."
        )
    return "\n".join(lines)


def error_examples(name: str, k: int = 6) -> str:
    path = RESULTS_DIR / f"errors_{name}.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path).head(k)
    if df.empty:
        return ""
    rows = ["| Avis | Réel | Prédit |", "|---|---|---|"]
    for _, row in df.iterrows():
        text = str(row["text"]).replace("|", "/").replace("\n", " ")
        text = text if len(text) <= 220 else text[:217] + "..."
        rows.append(f"| {text} | {row['réel']} | {row['prédit']} |")
    return "\n".join(rows)


def build_block(results: list[dict]) -> str:
    parts = ["## Résultats", "", "Résultats mesurés sur le jeu de test (avis jamais vus pendant l'entraînement).", ""]
    parts += [to_markdown(results), "", "![Comparaison des modèles](results/comparison.png)", ""]

    parts += ["### Observations", "", observations(results), ""]

    parts += ["### Matrices de confusion", "",
              "Chaque ligne = la vraie classe, chaque colonne = ce que le modèle a prédit. "
              "La diagonale est la part d'avis bien classés ; le reste montre *comment* le modèle se trompe.", ""]
    for r in results:
        img = f"results/confusion_matrix_{r['model']}.png"
        if _exists(img):
            parts += [f"**{NICE_NAMES.get(r['model'], r['model'])}**", "", f"![Matrice {r['model']}]({img})", ""]

    top = RESULTS_DIR / "top_words_baseline.md"
    if top.exists():
        parts += ["### Ce que la baseline a appris", "",
                  "Les mots les plus associés à chaque classe (coefficients de la régression logistique) :", "",
                  top.read_text(encoding="utf-8").strip(), ""]

    for r in results:
        table = error_examples(r["model"])
        if table:
            parts += [f"### Exemples d'erreurs : {NICE_NAMES.get(r['model'], r['model'])}", "", table, ""]

    parts += ["## Démo", ""]
    if _exists("docs/demo.png"):
        parts += ["![Démo Streamlit](docs/demo.png)", ""]
    else:
        parts += ["(Ajoute une capture de l'appli dans `docs/demo.png` puis relance ce script.)", ""]
    return "\n".join(parts)


def main() -> None:
    results = load_all()
    if not results:
        raise SystemExit("Aucun résultat dans results/. Lance d'abord baseline.py (et transformer.py).")
    plot(results)
    (RESULTS_DIR / "comparison.md").write_text(to_markdown(results) + "\n", encoding="utf-8")

    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit("Marqueurs RESULTS:START / RESULTS:END introuvables dans le README.")
    block = f"{START}\n{build_block(results)}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    README.write_text(new, encoding="utf-8")
    print("README mis à jour avec les résultats de results/.")


if __name__ == "__main__":
    main()
