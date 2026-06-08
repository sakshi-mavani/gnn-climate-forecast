import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph_builder import build_knn_graph


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

pos_dict = {
    i: positions[i]
    for i in range(len(positions))
}

plt.figure(figsize=(6, 5))

nx.draw(
    graph,
    pos=pos_dict,
    with_labels=True
)

plt.title("Sensor KNN Graph")

plt.savefig(
    "results/sensor_graph.png"
)

print("Graph saved successfully!")