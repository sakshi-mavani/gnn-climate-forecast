import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(node_positions, k=2):

    nbrs = NearestNeighbors(
        n_neighbors=k + 1
    )

    nbrs.fit(node_positions)

    distances, indices = nbrs.kneighbors(
        node_positions
    )

    G = nx.Graph()

    for i in range(len(node_positions)):
        G.add_node(i)

    for i, neighbors in enumerate(indices):

        for nbr in neighbors[1:]:

            G.add_edge(i, nbr)

    return G


if __name__ == "__main__":

    positions = np.array([
        [22.30, 70.78],
        [22.31, 70.79],
        [22.32, 70.77],
        [22.35, 70.80],
        [22.29, 70.76]
    ])

    graph = build_knn_graph(
        positions,
        k=2
    )

    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())

    print(
        list(graph.edges())
    )