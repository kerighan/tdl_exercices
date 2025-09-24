from datasets import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
from walker import random_walks
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np

# TODO charger le dataset MUTAG via la lib datasets de huggingface
edges = []
ds = load_dataset("graphs-datasets/MUTAG")

# TODO chaque élément de edge_index est un couple (sources, targets). Il faut manipuler les données pour obtenir 
# une liste edges = [(source, target) ...] unique pour tous les graphes (ie. graphes réunis au sein d'une seule liste edges)
# Aide : maintenir une liste "offsets" qui va incrémenter le nombre de noeuds de chaque graphe (garder les index de changement de graphe)
offsets = [0]
last_offset = 0
for (sources, targets) in ds["train"]["edge_index"]:
    for source, target in zip(sources, targets):
        edges.append((source + last_offset, target + last_offset))

    n_nodes = max(max(sources), max(targets)) + 1
    last_offset += n_nodes
    offsets.append(last_offset)

# TODO ici les élèves doivent créer un graphe networkx et visualiser: ils doivent voir plein de petits ilôts séparés. 
# NB: le centre peut sembler attacher sans l'être
G = nx.Graph()
G.add_edges_from(edges)
nx.draw(G, node_size=15)
plt.show()

# TODO continuer la liste edges en ajoutant un noeud virtuel connectant chaque noeud virtuel avec le reste du graphe
last_offset = offsets[-1]
n_virtual_nodes = 0
for i in range(len(offsets) - 1):
    virtual_node = i + last_offset
    start, end = offsets[i], offsets[i + 1]
    for j in range(start, end):
        edges.append((virtual_node, j))
    n_virtual_nodes += 1

# TODO recréer un graphe, et le visualiser en mettant en couleur les noeuds virtuels (peut se faire via [0, 0, ..., 1, 1])
G = nx.Graph()
G.add_edges_from(edges)
node_colors = np.array([0] * len(G.nodes))
node_colors[-n_virtual_nodes:] = 1
nodes = list(G.nodes)
nx.draw(G, node_size=15, node_color=node_colors)
plt.show()

# TODO vérifier que le nombre de composants connexes == 188
print(len(list(nx.connected_components(G))))

# TODO les élèves doivent réutiliser leur code de node2vec:
W = random_walks(G, n_walks=100, walk_len=21, p=0.25, q=0.25)
W = [row.tolist() for row in W]
model = Word2Vec(W, vector_size=128, window=8, min_count=0, sg=1, workers=8, epochs=10)
word_vectors = model.wv.vectors
id_to_node = dict(enumerate(model.wv.index_to_key))
node_to_id = {node: id for id, node in id_to_node.items()}
word_vectors = word_vectors[[node_to_id[node] for node in nodes]]

print(word_vectors.shape)
# TODO ne garder que les embeddings des noeuds virtuels (s'assurer qu'on garde bien 188 embeddings)
X = word_vectors[offsets[:-1]]
assert X.shape[0] == 188

# TODO extraire les labels y du dataset
y = ds["train"]["y"]
y = np.array([it[0] for it in y])

# TODO créer un petit classifieur
inp = tf.keras.layers.Input(shape=(X.shape[1]))
hidden = tf.keras.layers.Dense(24, activation="relu")(inp)
out = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)

model = tf.keras.models.Model(inp, out)
model.compile("adam", "binary_crossentropy", metrics=["binary_accuracy"])
model.fit(X, y, epochs=20, validation_split=.2)