# Analyse-de-l-Impact-des-D-penses-Publicitaires-sur-les-Ventes
📄Description du projet
Ce projet vise à analyser l'impact des dépenses publicitaires sur les ventes de produits, en utilisant des données sur les budgets publicitaires pour différents canaux (TV, radio, journaux). À travers l'utilisation de modèles de machine learning, nous cherchons à prédire les ventes et à identifier les canaux publicitaires les plus efficaces pour maximiser le retour sur investissement (ROI).

🎯 Objectifs du projet
Analyser la relation entre les dépenses publicitaires et les ventes pour comprendre quels canaux (TV, radio, journaux) ont le plus d'impact.
Construire des modèles prédictifs (Régression Linéaire et Forêt Aléatoire) pour estimer les ventes en fonction des dépenses.
Comparez les modèles afin de choisir le plus performant pour des prédictions fiables.
Fournir des recommandations pour l'optimisation des budgets publicitaires.
📊 Ensemble de Données
Colonnes :
TV: Dépenses publicitaires en milliers de dollars pour la télévision.
Radio: Dépenses publicitaires en milliers de dollars pour la radio.
Newspaper: Dépenses publicitaires en milliers de dollars pour les journaux.
Sales: Ventes en milliers d'unités (cible variable).
Source : Ensemble de données fictif basé sur des exemples classiques d'analyse publicitaire (disponible sur des plateformes comme Kaggle).
🛠️ Outils et Bibliothèques
Python : Langage utilisé pour l'analyse et la modélisation.
Pandas : Manipulation et analyse des données.
Scikit-Learn : Création et évaluation des modèles de machine learning.
Matplotlib et Seaborn : Visualisation des données et des résultats des modèles.
📈 Méthodologie
Analyse Exploratoire des Données (EDA) :

Visualisation de la répartition des budgets publicitaires pour chaque canal.
Calcul des corrélations entre les dépenses par canal et les ventes.
Modélisation Prédictive :

Régression Linéaire : Modèle simple et interprétable qui cherche à établir une relation linéaire entre les dépenses et les ventes.
Forêt Aléatoire : Modèle basé sur un ensemble d'arbres de décision pour capturer des relations non linéaires et complexes.
Comparaison des modèles :

Évaluation des modèles en utilisant le RMSE (Root Mean Squared Error) et le R² (coefficient de détermination).
Le modèle de Forêt Aléatoire a montré de meilleures performances en raison de sa capacité à capturer des interactions complexes.
🔍 Résultats et Interprétations
Régression Linéaire :

RMSE : 1,78
R² : 0,899
Ce modèle fonctionne bien pour des relations simples, mais manque de précision pour des interactions complexes.
Forêt Aléatoire :

RMSE : 0,768
R² : 0,981
Ce modèle s'est avéré plus performant pour capturer les relations complexes entre les variables publicitaires et les ventes.
📌 Recommandations
Réorienter les budgets vers les canaux les plus performants (notamment la TV, qui montre la plus forte corrélation avec les ventes).
Utiliser la Forêt Aléatoire pour les prévisions de ventes, car ce modèle offre une meilleure précision pour des relations non linéaires.
Surveiller et ajuster les dépenses publicitaires en fonction des résultats pour maximiser le retour sur investissement.
🚀 Conclusion
Ce projet montre comment une analyse de données et l'utilisation de modèles de machine learning peuvent optimiser les stratégies de dépenses publicitaires. La Forêt Aléatoire s'est révélée être le modèle le plus adapté pour prédire les ventes dans ce contexte, fournissant ainsi des recommandations utiles pour les décideurs en marketing.

📚 Références
Apprentissage automatique pratique avec Scikit-Learn et TensorFlow - Aurélien Géron (O'Reilly Media)
Documentation Scikit-Learn : Régression Linéaire , Forêt Aléatoire
