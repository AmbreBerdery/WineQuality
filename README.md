#Wine Quality : classification de la qualité du vin
Projet de Machine Learning 1 (ING2 APP BDML3, 2026-2027). Samy AIT ALLA, Ambre BERDERY, Camélia FIXARIS.
Objectif
Prédire si un vin est « bon » (note de qualité ≥ 7) à partir de onze mesures physicochimiques, en comparant cinq modèles de classification supervisée : Naive Bayes, régression logistique, arbre de décision, k plus proches voisins (k-NN) et SVM.
Les données sont les jeux Wine Quality rouge et blanc du dépôt UCI (Cortez et al., 2009) : 6 497 vins, 5 320 après suppression des doublons, dont 19 % de bons vins.
Résultat
Le modèle retenu est un SVM à noyau RBF (C ≈ 0,37, gamma ≈ 0,035, class_weight="balanced", 12 variables).
Modèle	Accuracy	Précision	Rappel	F1	AUC
SVM	0,728	0,399	0,851	0,543	0,842
k-NN	0,793	0,467	0,634	0,538	0,845
Régression logistique	0,712	0,373	0,752	0,498	0,804
Arbre de décision	0,709	0,366	0,733	0,488	0,784
Naive Bayes	0,693	0,352	0,738	0,477	0,772
Scores mesurés une seule fois sur le jeu de test (1 064 vins, dont 202 bons).
Structure du dépôt
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
Les notebooks 03 et 04 entraînent et optimisent chacun une partie des modèles, puis les sauvegardent dans models/. Le notebook 05 les charge tous et les compare sur le jeu de test : il doit donc être exécuté en dernier.
Méthode commune
Pour que les modèles soient comparables, tout le groupe utilise le même cadre, défini une seule fois dans src/ :
découpage train/test 80 % / 20 %, stratifié, random_state = 42 (src/load_data.py) ;
prétraitement dans un pipeline (plafonnement IQR, standardisation, sélection de variables), réajusté sur chaque pli pour éviter toute fuite de données (src/preprocess.py) ;
validation croisée StratifiedKFold(5, shuffle=True, random_state=42) et F1 comme métrique d'optimisation (src/tuning.py) ;
jeu de test utilisé une seule fois, au notebook 05.
Installation et exécution
Python 3.11 ou plus récent.
bash
git clone https://github.com/AmbreBerdery/WineQuality.git
cd WineQuality
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
Pour reproduire les résultats, au choix :
ouvrir WineQuality_notebook_final.ipynb et lancer Run All (environ 10 minutes) ;
ou exécuter les notebooks du dossier notebooks/ dans l'ordre : 01, 02, 03, 04, puis 05.
Lancer les tests unitaires :
bash
pytest
