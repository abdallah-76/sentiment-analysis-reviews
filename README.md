# 🛒 Analyse de sentiment d'avis clients e-commerce

> Un modèle qui lit un avis produit et dit s'il est **négatif, neutre ou positif**.
> Deux approches comparées sur les mêmes données : un modèle classique rapide (**TF-IDF + régression logistique**) et un transformer pré-entraîné (**DistilBERT**).

**Stack :** Python · pandas · scikit-learn · Hugging Face Transformers · PyTorch · Matplotlib · Streamlit

## Sommaire

1. [Le problème](#le-problème)
2. [Les données](#les-données)
3. [La démarche](#la-démarche)
4. [Les deux modèles en détail](#les-deux-modèles-en-détail)
5. [Comment lire les métriques](#comment-lire-les-métriques)
6. [Installation et utilisation](#installation-et-utilisation)
7. [Résultats](#résultats)
8. [Structure du projet](#structure-du-projet)
9. [Choix techniques](#choix-techniques)
10. [Limites et pistes d'amélioration](#limites-et-pistes-damélioration)

---

## Le problème

Une boutique en ligne reçoit des milliers d'avis. Impossible de tous les lire, mais c'est là que se cachent les produits défectueux, les problèmes de livraison ou les points forts à mettre en avant.

Ce projet classe automatiquement chaque avis pour **repérer rapidement ce qui va mal** et prioriser le service client. La question de fond : *est-ce qu'un gros modèle de langage fait vraiment mieux qu'une méthode simple, et à quel coût ?*

## Les données

- **Source :** avis Amazon en anglais, miroir [`SetFit/amazon_reviews_multi_en`](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en) sur Hugging Face (environ 200 000 avis d'entraînement, 5 000 de validation, 5 000 de test).
- **Transformation :** les notes de 1 à 5 étoiles sont regroupées en 3 classes.

| Étoiles | Classe |
|---|---|
| 1-2 ★ | négatif |
| 3 ★ | neutre |
| 4-5 ★ | positif |

Le jeu de données est équilibré par note, donc les classes sont réparties environ **40 % / 20 % / 40 %**. C'est pour cela qu'on suit le **F1 macro** et pas seulement l'accuracy.

## La démarche

```
                 ┌─► TF-IDF + Régression logistique  (baseline : rapide, interprétable)
Avis brut ───────┤
                 └─► DistilBERT fine-tuné            (transformer pré-entraîné)
                                   │
                                   ▼
        Même évaluation : accuracy, F1 macro, matrice de confusion, analyse d'erreurs
```

1. **Préparation** : regroupement des notes en 3 classes, nettoyage minimal (suppression des avis vides).
2. **Baseline** : on entraîne d'abord le modèle le plus simple possible. Il sert de point de comparaison : un modèle complexe n'a d'intérêt que s'il fait nettement mieux.
3. **DistilBERT** : fine-tuning d'un modèle pré-entraîné.
4. **Comparaison** sur le même jeu de test : qualité, temps d'entraînement, types d'erreurs.

## Les deux modèles en détail

### Baseline : TF-IDF + régression logistique

- **TF-IDF** transforme chaque avis en vecteur : un mot compte d'autant plus qu'il est fréquent dans l'avis *et* rare dans l'ensemble des avis. Les mots vides (« the », « and ») pèsent peu, les mots marquants (« refund », « perfect ») pèsent beaucoup.
- Les **paires de mots** (bigrammes) sont incluses pour capter des tournures comme « not good » ou « works great ».
- La **régression logistique** apprend un poids par mot et par classe. On peut donc *lire* le modèle : quels mots poussent vers « négatif » ? (voir [Résultats](#résultats)).
- `class_weight="balanced"` compense le fait que la classe « neutre » est deux fois plus rare.

### DistilBERT fine-tuné

- DistilBERT est une version allégée de BERT : plus petit et plus rapide, il garde l'essentiel des performances. Il a déjà été pré-entraîné sur une grande quantité de textes anglais, il « comprend » donc le contexte d'un mot dans une phrase (« not good » ≠ « good »).
- **Fine-tuning** : on ajoute une couche de classification à 3 sorties et on réentraîne légèrement l'ensemble sur nos avis (learning rate 2e-5, 1 epoch par défaut). C'est du *transfer learning* : on réutilise ce que le modèle sait déjà au lieu de repartir de zéro.
- Les avis sont tronqués à 128 tokens et le padding est dynamique pour accélérer l'entraînement.

## Comment lire les métriques

| Métrique | Ce qu'elle dit |
|---|---|
| **Accuracy** | Part d'avis correctement classés. Simple, mais trompeuse si les classes sont déséquilibrées. |
| **F1 par classe** | Compromis entre précision (« quand je prédis X, ai-je raison ? ») et rappel (« est-ce que je trouve tous les X ? »). |
| **F1 macro** | Moyenne des F1 des trois classes, sans pondération : un modèle qui rate la classe « neutre » est pénalisé. C'est la métrique principale ici. |
| **Matrice de confusion** | Montre *quelles* classes sont confondues entre elles. |

## Installation et utilisation

```bash
git clone <url-du-repo>
cd sentiment-analysis-reviews
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# 1. Baseline (environ 1 minute sur 20 000 avis)
python src/baseline.py --train-size 20000

# 2. DistilBERT (lent sur CPU, voir remarque)
python src/transformer.py --train-size 20000

# 3. Comparaison : tableau + graphique dans results/
python src/compare.py

# 4. Démo interactive
streamlit run app.py

# 5. (après avoir mis une capture de la démo dans docs/demo.png) remplir ce README avec les résultats
python src/update_readme.py
```

**Remarques**
- Utiliser le **même `--train-size`** pour les deux modèles donne une comparaison équitable. La baseline peut aussi tourner sur les ~200 000 avis (sans `--train-size`).
- Le fine-tuning de DistilBERT est **lent sur CPU**. Pour aller vite : GPU gratuit sur Google Colab, ou `--train-size` plus petit.
- Pas d'accès à Hugging Face ? Utilise ton propre CSV : `--csv mes_avis.csv` (colonnes `text` et `label`, label = 0/1/2 ou négatif/neutre/positif).

<!-- RESULTS:START -->
## Résultats

Résultats mesurés sur le jeu de test (avis jamais vus pendant l'entraînement).

| Modèle | Avis d'entraînement | Accuracy | F1 macro | F1 négatif | F1 neutre | F1 positif | Temps d'entraînement |
|---|---|---|---|---|---|---|---|
| TF-IDF + Régression logistique | 20 000 | 68.2% | 0.631 | 0.746 | 0.366 | 0.780 | 7 s |
| DistilBERT (fine-tuné) | 20 000 | 74.2% | 0.664 | 0.796 | 0.355 | 0.842 | 91.0 min |

![Comparaison des modèles](results/comparison.png)

### Observations

- Meilleur F1 macro : **DistilBERT (fine-tuné)** (0.664, accuracy 74.2%).
- Écart de F1 macro entre les deux modèles : **0.034** (20 000 vs 20 000 avis d'entraînement).
- Temps d'entraînement : DistilBERT (fine-tuné) 91.0 min contre 7 s pour TF-IDF + Régression logistique (environ x780).
- TF-IDF + Régression logistique : classe la plus difficile = **neutre** (F1 0.366).
- DistilBERT (fine-tuné) : classe la plus difficile = **neutre** (F1 0.355).

### Matrices de confusion

Chaque ligne = la vraie classe, chaque colonne = ce que le modèle a prédit. La diagonale est la part d'avis bien classés ; le reste montre *comment* le modèle se trompe.

**TF-IDF + Régression logistique**

![Matrice baseline](results/confusion_matrix_baseline.png)

**DistilBERT (fine-tuné)**

![Matrice distilbert](results/confusion_matrix_distilbert.png)

### Ce que la baseline a appris

Les mots les plus associés à chaque classe (coefficients de la régression logistique) :

| Classe | Mots les plus associés |
|---|---|
| négatif | not, disappointed, even, never, terrible, broke, return, not worth, useless, horrible |
| neutre | but, however, okay, not very, ok, though, three stars, otherwise, meh, membership |
| positif | great, love, perfect, good, not too, perfectly, nice, excellent, loves, easy |

### Exemples d'erreurs : TF-IDF + Régression logistique

| Avis | Réel | Prédit |
|---|---|---|
| These are AWFUL. They are see through, the fabric feels like tablecloth, and they fit like children’s clothing. Customer service did seem to be nice though, but I regret missing my return date for these. I wouldn’t ev... | négatif | neutre |
| Really, really thin. | négatif | positif |
| My package arrived with nothing in it?! My cord that I use for my phone is broken and I was really depending on this replacement. I will give a better rating when I have something to rate!! | négatif | neutre |
| I've been using this product with my laundry for a while now and I like it, but I had to return it because it wasn't packaged right and everything was damaged. | négatif | neutre |
| Added so much to the projects! Works well. | négatif | positif |
| I have been using this for several years and recently bought several only to find out that they all have perfume in them now. | négatif | positif |

### Exemples d'erreurs : DistilBERT (fine-tuné)

| Avis | Réel | Prédit |
|---|---|---|
| Added so much to the projects! Works well. | négatif | positif |
| I have been using this for several years and recently bought several only to find out that they all have perfume in them now. | négatif | positif |
| When you order a product, you expect the product to be as described. They were advertised as CAP Dumbbells. However, the product received was Golds Gym dumbbells. If I wanted Golds Gym, I would have bought these from ... | négatif | positif |
| Doesn't distribute the steam strong enough small fan but it is quiet but who cares | négatif | neutre |
| Use at your own risk. My wife and I both had burning sensation and skin reactions. No history of sensitive skin. | négatif | positif |
| I love it It is adorable , perfect size! But my glass part has a crack, can you please replace the glass part? Thank you | négatif | positif |

## Démo

![Démo Streamlit](docs/demo.png)

<!-- RESULTS:END -->
![Interface de la démo](docs/interface.png)

## Analyse

À nombre d'avis égal (20 000), DistilBERT dépasse la baseline en accuracy (74,2 % contre 68,2 %), surtout sur les classes négative et positive. L'écart de F1 macro est plus modeste (environ 0.03), car la classe « neutre » reste difficile pour les deux modèles (F1 autour de 0.36) : les avis à 3 étoiles mélangent des signaux positifs et négatifs.

Le gain de DistilBERT a un coût : 91 minutes d'entraînement sur CPU contre 7 secondes pour la baseline.

Les exemples d'erreurs montrent aussi un problème de données : plusieurs avis « mal classés » ont un texte clairement positif (« Works well », « I love it... perfect size ») mais une note de 1 ou 2 étoiles. Une partie des erreurs vient donc du label, pas du modèle.

Piste non testée : DistilBERT n'a pas de pondération de classes (contrairement à la baseline) et n'a fait qu'une epoch, ce qui peut expliquer en partie son rappel plus faible sur le neutre.

## Structure du projet

```
sentiment-analysis-reviews/
├── app.py                # démo Streamlit
├── requirements.txt
├── docs/
│   └── demo.png          # capture de la démo
├── src/
│   ├── data.py           # chargement + regroupement des notes en 3 classes
│   ├── baseline.py       # TF-IDF + régression logistique
│   ├── transformer.py    # fine-tuning de DistilBERT
│   ├── metrics.py        # métriques, matrice de confusion, erreurs
│   ├── compare.py        # tableau et graphique de comparaison
│   └── update_readme.py  # remplit la section Résultats de ce README
└── results/              # métriques, graphiques, erreurs (générés)
```

## Choix techniques

- **Baseline d'abord** : toujours partir du modèle le plus simple pour savoir ce que la complexité apporte réellement.
- **F1 macro plutôt qu'accuracy** : classes déséquilibrées (40/20/40).
- **Même jeu de test, mêmes métriques** pour les deux modèles (`metrics.py`), afin que la comparaison soit honnête.
- **Résultats sauvegardés en JSON/CSV/PNG** : tout est reproductible, et le README est rempli depuis ces fichiers plutôt qu'à la main.
- **Graine fixée (`--seed 42`)** pour des résultats reproductibles.

## Limites et pistes d'amélioration

- **Le label vient de la note, pas du texte** : un avis à 3 étoiles peut être très positif dans les mots, et la classe « neutre » est naturellement la plus floue.
- **Anglais uniquement.** Étape suivante : un modèle multilingue ou français (CamemBERT) pour des avis francophones.
- **Courbe d'apprentissage** : tracer la performance en fonction du nombre d'avis pour voir à partir de quand DistilBERT dépasse la baseline.
- **Aller plus loin** : analyser *quel aspect* est jugé (livraison, prix, qualité), ou déployer la démo en ligne.

---

Projet réalisé par Abdallah Maammar, Master Intelligence Artificielle (Université Toulouse III).
