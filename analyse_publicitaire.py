
# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Chargement des données
data = pd.read_csv('Advertising.csv')  # Remplacez par le chemin de votre fichier CSV

# Affichage des premières lignes des données pour vérification
print(data.head())

# ### 1. Analyse Exploratoire des Données (EDA) ###

# Visualisation des distributions de dépenses par canal
plt.figure(figsize=(18, 5))

# Distribution des dépenses TV
plt.subplot(1, 3, 1)
sns.histplot(data['TV'], kde=True, color="skyblue")
plt.title("Distribution des Dépenses TV")
plt.xlabel("TV")
plt.ylabel("Count")

# Distribution des dépenses Radio
plt.subplot(1, 3, 2)
sns.histplot(data['radio'], kde=True, color="olive")
plt.title("Distribution des Dépenses Radio")
plt.xlabel("Radio")
plt.ylabel("Count")

# Distribution des dépenses Journaux
plt.subplot(1, 3, 3)
sns.histplot(data['newspaper'], kde=True, color="gold")
plt.title("Distribution des Dépenses Journaux")
plt.xlabel("Journaux")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Visualisation de la corrélation entre les variables
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Corrélation entre les Dépenses Publicitaires et les Ventes")
plt.show()

# Diagrammes de dispersion pour chaque canal publicitaire vs ventes
plt.figure(figsize=(18, 5))

# TV vs Ventes
plt.subplot(1, 3, 1)
sns.scatterplot(x='TV', y='sales', data=data, color="skyblue")
plt.title("Dépenses TV vs Ventes")
plt.xlabel("TV")
plt.ylabel("Ventes")

# Radio vs Ventes
plt.subplot(1, 3, 2)
sns.scatterplot(x='radio', y='sales', data=data, color="olive")
plt.title("Dépenses Radio vs Ventes")
plt.xlabel("Radio")
plt.ylabel("Ventes")

# Journaux vs Ventes
plt.subplot(1, 3, 3)
sns.scatterplot(x='newspaper', y='sales', data=data, color="gold")
plt.title("Dépenses Journaux vs Ventes")
plt.xlabel("Journaux")
plt.ylabel("Ventes")

plt.tight_layout()
plt.show()

# ### 2. Préparation des Données ###

# Séparation des variables indépendantes et de la variable cible
X = data[['TV', 'radio', 'newspaper']]
y = data['sales']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### 3. Modélisation ###

# Modèle de Régression Linéaire
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Modèle de Forêt Aléatoire
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)

# ### 4. Évaluation des Modèles ###

# Fonction pour afficher les résultats de performance
def evaluation_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Évaluation de la Régression Linéaire
evaluation_metrics(y_test, y_pred_linear, "Régression Linéaire")

# Évaluation de la Forêt Aléatoire
evaluation_metrics(y_test, y_pred_forest, "Forêt Aléatoire")

# ### 5. Visualisation des Résultats ###

# Comparaison des prédictions de la Régression Linéaire vs Ventes Réelles
plt.figure(figsize=(12, 6))

# Régression Linéaire
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, color="blue", label="Prédictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color="red", label="Idéal")
plt.xlabel("Ventes Réelles")
plt.ylabel("Ventes Prédites")
plt.title("Régression Linéaire : Ventes Réelles vs Prédites")
plt.legend()

# Forêt Aléatoire
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_forest, color="green", label="Prédictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color="red", label="Idéal")
plt.xlabel("Ventes Réelles")
plt.ylabel("Ventes Prédites")
plt.title("Forêt Aléatoire : Ventes Réelles vs Prédites")
plt.legend()

plt.tight_layout()
plt.show()
