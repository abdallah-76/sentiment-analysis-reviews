"""Fine-tuning de DistilBERT pour la classification de sentiment (3 classes).

L'idée (transfer learning) : DistilBERT a déjà "lu" des millions de textes anglais.
On ne repart pas de zéro, on l'ajuste simplement à notre tâche avec quelques milliers d'avis.

Usage :
    python src/transformer.py                        # 10 000 avis, 1 epoch
    python src/transformer.py --train-size 20000     # comparaison à données égales avec la baseline
    python src/transformer.py --csv mes_avis.csv

Sur CPU c'est lent (comptez de l'ordre de plusieurs dizaines de minutes pour 10 000 avis).
Avec un GPU gratuit (Google Colab) cela prend quelques minutes.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import LABELS, ROOT, load_splits
from metrics import evaluate


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--csv", help="CSV local (colonnes text, label) à la place du dataset Hugging Face")
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128, help="Longueur max d'un avis en tokens")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train, val, test = load_splits(args.csv, args.train_size, args.seed)
    print(f"Train : {len(train)} | Validation : {len(val)} | Test : {len(test)}")
    print(f"Appareil : {'GPU' if torch.cuda.is_available() else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def to_dataset(df):
        ds = Dataset.from_pandas(df, preserve_index=False)
        return ds.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length),
            batched=True,
            remove_columns=["text"],
        )

    train_ds, val_ds, test_ds = to_dataset(train), to_dataset(val), to_dataset(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={name: i for i, name in enumerate(LABELS)},
    )

    training_args = TrainingArguments(
        output_dir=str(ROOT / "models" / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start
    print(f"Entraînement terminé en {train_seconds / 60:.1f} min")

    output = trainer.predict(test_ds)
    y_pred = np.argmax(output.predictions, axis=-1)
    evaluate("distilbert", test["text"], test["label"], y_pred, train_seconds, n_train=len(train))

    save_dir = ROOT / "models" / "distilbert"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"Modèle sauvegardé dans {save_dir}")


if __name__ == "__main__":
    main()
