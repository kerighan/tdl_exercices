# **Exercice : Embedding et classification de graphes (MUTAG)**

**Contexte** : Le dataset **MUTAG** est une collection de 188 molécules étiquetées selon leur activité mutagène (positif/négatif). L’objectif est de prédire cette activité à partir de la structure des molécules, représentées sous forme de graphes.

---

## **1. Chargement et préparation des données**

* Chargez le dataset **MUTAG** via `load_dataset("graphs-datasets/MUTAG")` issue de la librairie `datasets` de huggingface.
* Construisez une liste `edges` contenant toutes les arêtes de tous les graphes, en utilisant une liste `offsets` pour mémoriser les décalages d’index entre graphes.
* Créez un graphe `G` avec `networkx` et visualisez-le. **Question** : Que remarquez-vous sur la structure du graphe ?

---

## **2. Ajout de nœuds virtuels**

* Ajoutez un nœud virtuel par graphe, connecté à tous les nœuds du graphe correspondant.
* Visualisez à nouveau le graphe en colorant les nœuds virtuels (ex. : `[0, 0, ..., 1, 1]`).
* Vérifiez que le nombre de composantes connexes est égal à **188**.

---

## **3. Génération d’embeddings**

* Utilisez votre implémentation de **Node2Vec** pour générer des séquences de marches aléatoires sur le graphe modifié.
* Apprenez des embeddings de taille 128 avec `Word2Vec`.
* Extrayez uniquement les embeddings des **nœuds virtuels** (188 embeddings).

---

## **4. Classification**

* Extrayez les labels `y` depuis le dataset pour en faire un vecteur binaire
* Construisez un classifieur simple :
  * Couche d’entrée : taille des embeddings.
  * Couche cachée : 24 neurones (ReLU).
  * Couche de sortie : 1 neurone (sigmoïde).
* Entraînez le modèle avec `binary_crossentropy` et 20% de validation. **Question** : Quelle précision obtenez-vous ?
