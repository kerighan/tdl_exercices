# **Exercice : Embedding de graphe et classification nodale**

## **1. Chargement et visualisation**

* Chargez le graphe `G` depuis `cora/G.gexf` et les labels depuis `cora/labels.npy`.
* Visualisez le graphe à l’aide de `nx.draw`.
* **Question** : Observez-vous de l’homophilie ? Justifiez.

---

## **2. Génération de séquences**

### **2.1. Random walks classiques**

* Implémentez une fonction `random_walks(G, n_walks, walk_len)` pour générer des marches aléatoires.

### **2.2. Node2Vec**

* Implémentez `node2vec_walks(G, n_walks, walk_len, p, q)` avec les probabilités de transition de Node2Vec.

---

## **3. Embeddings avec Word2Vec**

* Apprenez des embeddings de taille 128 avec `Word2Vec` de la librairie Gensim (pip install gensim)

---

## **4. Classification**

* Construisez un réseau simple :
  * Couche d’entrée : 128 neurones.
  * Couche cachée : 64 neurones (ReLU).
  * Couche de sortie : `softmax` (nombre de neurones = nombre de classes).
* Entraînez le modèle avec `sparse_categorical_crossentropy` et 20% de validation.
* **Question** : Quelle précision obtenez-vous ?

---

## **5. Analyse**

* Testez différentes valeurs de `p` et `q` pour Node2Vec.
* Comparez les performances des deux méthodes.
* Proposez une amélioration possible.