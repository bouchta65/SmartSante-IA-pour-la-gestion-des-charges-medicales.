# SmartSante IA

## Contexte du projet
SmartSante IA est un projet développé dans le cadre d'une compagnie d’assurance santé visant à mettre en place un système intelligent capable d’estimer les charges médicales que chaque assuré devra payer.  

L’objectif est de proposer une solution innovante permettant d’anticiper les coûts, d’améliorer la transparence pour les assurés et d’aider l’entreprise à mieux gérer ses politiques tarifaires.

Les données utilisées proviennent des dossiers médicaux et incluent les critères suivants :  
- Âge (`Age`)  
- Sexe (`Sex`)  
- Indice de masse corporelle (`BMI`)  
- Nombre d’enfants à charge (`Children`)  
- Habitude tabagique (`Smoker`)  
- Région géographique (`Region`)  

---

## Features / Stories

### Feature Story 1 : Analyse et Préparation des Données
**Objectif** : Préparer les données pour garantir qu'elles sont propres, cohérentes et prêtes pour l'entraînement du modèle.  

**Tâches :**  
1. Chargement des données avec Pandas et vérification des types et structures des colonnes.  
2. Analyse exploratoire des données (EDA) :  
   - Aperçu des données et statistiques descriptives.  
   - Identification des valeurs manquantes et des doublons.  
   - Analyse de la distribution des variables numériques et relations entre variables (heatmaps, pairplots).  
3. Prétraitement :  
   - Gestion des valeurs manquantes.  
   - Suppression des doublons et traitement des outliers.  
   - Encodage des variables catégoriques.  
   - Division en ensembles d’entraînement et de test (80%/20%).  
   - Normalisation ou standardisation des variables numériques.  

### Feature Story 2 : Entraînement des Modèles de Régression
**Objectif** : Entraîner plusieurs modèles pour prédire les charges et évaluer leurs performances.  

**Tâches :**  
- Modèles utilisés : LinearRegression, RandomForestRegressor, XGBRegressor, SVR.  
- Utilisation de pipelines Scikit-learn intégrant prétraitement et entraînement.  
- Évaluation avec RMSE, MAE et R².  

### Feature Story 3 : Tuning des Hyperparamètres
**Objectif** : Optimiser les hyperparamètres des modèles les plus performants.  

**Tâches :**  
- Sélection des modèles performants.  
- Utilisation de GridSearchCV ou RandomizedSearchCV avec validation croisée (5 folds).  
- Réentraîner les modèles optimisés sur l’ensemble d’entraînement complet.  

### Feature Story 4 : Évaluation et Comparaison des Modèles
**Objectif** : Sélectionner le meilleur modèle basé sur les performances et la robustesse.  

**Tâches :**  
- Visualisations comparatives (résidus, prédictions vs. réelles).  
- Tableau récapitulatif des performances (RMSE, MAE, R²).  

### Feature Story 5 : Test et Téléchargement du Modèle
**Objectif** : Déployer le modèle et créer une interface utilisateur simple.  

**Tâches :**  
- Export du modèle avec `joblib.dump()` ou `pickle`.  
- Interface utilisateur permettant de saisir les données et obtenir une estimation des charges.  
