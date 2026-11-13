# Wine Quality : classification de la qualité du vin

Projet de Machine Learning 1 (ING2 APP BDML3, 2026-2027).
Samy AIT ALLA, Ambre BERDERY, Camélia FIXARIS.

## Objectif

Prédire si un vin est « bon » (note de qualité ≥ 7) à partir de onze mesures physicochimiques, en comparant cinq modèles de classification supervisée : Naive Bayes, régression logistique, arbre de décision, k plus proches voisins (k-NN) et SVM.

Les données sont les jeux Wine Quality rouge et blanc du dépôt UCI (Cortez et al., 2009) : 6 497 vins, 5 320 après suppression des doublons, dont 19 % de bons vins.


## Structure du dépôt

```
WineQuality/
├── WineQuality_notebook_final.ipynb   notebook unique du rendu (les 5 parties, exécuté)
├── data/raw/                          winequality-red.csv, winequality-white.csv
├── src/
│   ├── load_data.py                   chargement, cible binaire, découpage train/test
│   ├── preprocess.py                  nettoyage, IQRClipper, pipeline de prétraitement
│   ├── feature_selection.py           filtre par corrélation, SelectKBest, RFE
│   ├── models.py                      les cinq modèles et leurs grilles d'hyperparamètres
│   ├── tuning.py                      validation croisée commune, Grid Search, Random Search
│   └── evaluate.py                    métriques, matrices de confusion, ROC, biais/variance
├── notebooks/
│   ├── 01_exploration.ipynb               analyse exploratoire
│   ├── 02_pretraitement.ipynb             nettoyage, pipeline et sélection de variables
│   ├── 03_modeles_nb_logreg_arbre.ipynb   Naive Bayes, régression logistique, arbre de décision
│   ├── 04_modeles_knn_svm.ipynb           k-NN, SVM, comparaison Grid Search / Random Search
│   └── 05_evaluation.ipynb                évaluation finale des cinq modèles sur le jeu de test
├── tests/                             tests unitaires (pytest)
└── models/                            modèles sauvegardés (générés, non versionnés)
```
