import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

class Graph():
    def __init__(self, n):
        self.nodes = n
        self.adj = np.zeros((n, n))
        self.edges = {}
        self.degrees = {i : 0 for i in range(n)}

    def init_er(self, p, v = None):
        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                if random.uniform(0, 1.0) < p:
                    if v:
                        self.add_edge(i , j, v)
                    else:
                        self.add_edge(i, j)

    def add_edge(self, i, j, v = None):
        self.adj[i][j] = 1
        self.adj[j][i] = 1
        self.degrees[i] += 1
        self.degrees[j] += 1
        if v:
            self.edges[(i,j)] = random.uniform(0, 1)
        else:
            self.edges[(i,j)] = 1

    def get_degrees(self):
        return self.degrees.copy()

    def get_adj_matrix(self):
        return self.adj[:][:]

    def get_edge_list(self):
        return self.edges.copy()
    
    def __str__(self):
        return str(self.edges.keys())

    def visualize(self):
        G = nx.Graph()
        G.add_nodes_from([i for i in range(self.nodes)])
        G.add_edges_from(self.edges.keys())
        nx.draw_networkx(G, with_labels=False)
        plt.show()
    
    def visualize_with_weights(self):
        G = nx.Graph()
        for edge, weight in self.edges.items():
            G.add_edge(edge[0], edge[1], weight=weight)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

if __name__ == "__main__":
    g = Graph(50)
    g.init_er(0.2)
    g.visualize()