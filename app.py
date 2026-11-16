"""Démo interactive : colle un avis, le modèle prédit son sentiment.

Lancer avec :  streamlit run app.py
(entraîne d'abord au moins la baseline : python src/baseline.py)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
LABELS = ["négatif", "neutre", "positif"]
BASELINE_PATH = ROOT / "models" / "baseline.joblib"
BERT_PATH = ROOT / "models" / "distilbert"

EXAMPLES = {
    "Positif": "Great quality for the price, arrived fast and works exactly as described. Would buy again!",
    "Neutre": "It does the job. Nothing special, the battery is okay but the build feels a bit cheap.",
    "Négatif": "Stopped working after two days. Support never answered my emails. Complete waste of money.",
}


@st.cache_resource
def load_baseline():
    return joblib.load(BASELINE_PATH) if BASELINE_PATH.exists() else None


@st.cache_resource
def load_bert():
    if not BERT_PATH.exists():
        return None
    from transformers import pipeline

    return pipeline("text-classification", model=str(BERT_PATH), top_k=None)


def predict_baseline(model, text: str) -> pd.Series:
    return pd.Series(model.predict_proba([text])[0], index=LABELS)


def predict_bert(model, text: str) -> pd.Series:
    out = model(text, truncation=True, max_length=128)
    if isinstance(out[0], list):  # selon la version de transformers
        out = out[0]
    scores = {item["label"]: item["score"] for item in out}
    return pd.Series([scores[label] for label in LABELS], index=LABELS)


st.set_page_config(page_title="Sentiment des avis clients", page_icon="🛒")
st.title("🛒 Sentiment d'un avis client")
st.caption("Colle un avis produit (en anglais) : le modèle dit s'il est négatif, neutre ou positif.")

baseline, bert = load_baseline(), load_bert()
available = {}
if baseline is not None:
    available["TF-IDF + Régression logistique"] = lambda t: predict_baseline(baseline, t)
if bert is not None:
    available["DistilBERT"] = lambda t: predict_bert(bert, t)

if not available:
    st.error("Aucun modèle trouvé. Lance d'abord `python src/baseline.py` (et `python src/transformer.py`).")
    st.stop()

choice = st.selectbox("Modèle", list(available))
example = st.selectbox("Exemple", ["(écrire mon propre avis)"] + list(EXAMPLES))
default = EXAMPLES.get(example, "")
text = st.text_area("Avis", value=default, height=140)

if st.button("Analyser") and text.strip():
    probs = available[choice](text)
    st.subheader(f"Prédiction : {probs.idxmax()} ({probs.max():.0%})")
    st.bar_chart(probs)
