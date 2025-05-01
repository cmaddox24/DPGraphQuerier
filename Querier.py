import numpy as np
from Graph import Graph
import networkx as nx

class Querier():
    def __init__(self, G):
        self.g = G
    
    def max_degree(self):
        deg_max = 0
        for deg in self.g.get_degrees().values():
            deg_max = max(deg_max, deg)
        return deg_max

    def min_degree(self):
        deg_min = self.g.nodes + 1
        for deg in self.g.get_degrees().values():
            deg_min = min(deg_min, deg)
        return deg_min

    def avg_degree(self):
        total = 0
        for deg in self.g.get_degrees().values():
            total += deg
        return total/self.g.nodes

    def degree_hist(self):
        hist = [0 for i in range(self.g.nodes)]
        for deg in self.g.get_degrees().values():
            hist[deg] += 1
        return hist

    def triangle_count(self):
        nx_g = nx.Graph()
        nx_g.add_edges_from(self.g.get_edge_list().keys())
        return sum(nx.triangles(nx_g).values()) // 3

    def average_edge_val(self):
        edges = self.g.get_edge_list()
        return sum(edges.values())/len(edges)

class Lap_Querier(Querier):
    pass

class LP_Querier(Querier):
    pass

class Proj_Querier(Querier):
    pass

class Gen_Querier(Querier):
    pass

if __name__=="__main__":
    g = Graph(5)
    g.init_er(0.7, [0, 1])
    q = Querier(g)
    queries = [q.min_degree, q.max_degree, q.avg_degree, q.degree_hist, q.triangle_count, q.average_edge_val]
    print(g)
    for query in queries:
        print(query())
    g.visualize_with_weights()