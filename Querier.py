import numpy as np
from Graph import Graph
import networkx as nx
import math
import matplotlib.pyplot as plt

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

class Node_Lap_Querier(Querier):
    def __init__(self, g, ep):
        super().__init__(g)
        self.ep = ep
    
    def _sample_lap(self, param):
        return np.random.laplace(0, param, 1)[0]
    
    def max_degree(self):
        real = super().max_degree()
        gsf = self.g.nodes
        param = gsf/self.ep
        return round(real + self._sample_lap(param))

    def min_degree(self):
        real = super().min_degree()
        gsf = self.g.nodes
        param = gsf/self.ep
        return round(real + self._sample_lap(param))

    def avg_degree(self):
        real = super().avg_degree()
        gsf = 1
        param = gsf/self.ep
        return real + self._sample_lap(param)

    def degree_hist(self):
        real = super().degree_hist()
        gsf = self.g.nodes
        param = gsf/self.ep
        return [round(i + self._sample_lap(param)) for i in real]

    def triangle_count(self):
        real = super().triangle_count()
        gsf = math.comb(self.g.nodes - 1, 2)
        param = gsf/self.ep
        return round(real +self._sample_lap(param))

    def average_edge_val(self):
        real = super().average_edge_val()
        gsf = 1
        param = gsf/self.ep
        return real + self._sample_lap(param)

class Edge_Lap_Querier(Querier):
    def __init__(self, g, ep):
        super().__init__(g)
        self.ep = ep
    
    def _sample_lap(self, param):
        return np.random.laplace(0, param, 1)[0]
    
    def max_degree(self):
        real = super().max_degree()
        gsf = 1
        param = gsf/self.ep
        return round(real + self._sample_lap(param))

    def min_degree(self):
        real = super().min_degree()
        gsf = 1
        param = gsf/self.ep
        return round(real + self._sample_lap(param))

    def avg_degree(self):
        real = super().avg_degree()
        gsf = 1
        param = gsf/self.ep
        return real + self._sample_lap(param)

    def degree_hist(self):
        real = super().degree_hist()
        gsf = 1
        param = gsf/self.ep
        return [round(i + self._sample_lap(param)) for i in real]

    def triangle_count(self):
        real = super().triangle_count()
        gsf = 2
        param = gsf/self.ep
        return round(real +self._sample_lap(param))

    def average_edge_val(self):
        real = super().average_edge_val()
        gsf = 1
        param = gsf/self.ep
        return real + self._sample_lap(param)
    
class Graph_Gen_Querier(Querier):
    def __init__(self, g, ep):
        super().__init__(g)
        k1_dist = self.degree_hist()
        k1_private = [round(i + self._sample_lap(4/ep)) for i in k1_dist]
        deg_seq = []
        for i in range(len(k1_private)):
            for _ in range(k1_private[i]):
                deg_seq.append(i)
        g1 = nx.expected_degree_graph(deg_seq, selfloops=False)
        nodes = len(g1.nodes())
        self.g = Graph(nodes)
        for edge in g1.edges():
            self.g.add_edge(edge[0], edge[1], True)
    
    def viz_gen_graph(self):
        self.g.visualize()

    def _sample_lap(self, param):
        return np.random.laplace(0, param, 1)[0]

class Proj_Querier(Querier):
    def __init__(self, g, ep1, ep2, O):
        super().__init__(g)
        self.ep2 = ep2
        scores = {}
        hist = super().degree_hist()
        for i in range(11):
            buckets = []
            ex = 0
            while sum(buckets) < g.nodes:
                buckets.append(math.floor((1 + i / 10)**ex))
                if sum(buckets) > g.nodes:
                    buckets[-1] = g.nodes - sum(buckets[:-1])
                ex += 1
            for j in range(1, O):
                score = 0
                cur_degree = 0
                for bucket in buckets:
                    for l in range(bucket):
                        score += (abs(hist[cur_degree] - (hist[cur_degree]/bucket)) + (2*j + 1)/ (bucket*(ep1 + ep2))) #lhist
                        score += 2 * (sum(hist[cur_degree+1:])) #lproj
                        cur_degree += 1
                scores[(i, j)] = self._sample_exp((ep1*score)/2*(6*O + 4))
        
        min_score = math.inf
        min_pair = None
        for pair, score in scores.items():
            if score < min_score:
                min_score = score
                min_pair = pair

        sigma, theta = min_pair
        self.theta = theta
        self.hist_buckets = []
        ex = 0
        while sum(self.hist_buckets) < g.nodes:
            self.hist_buckets.append(math.floor((1 + sigma / 10)**ex))
            ex += 1
        proj_g = Graph(g.nodes)
        degrees = {i : 0 for i in range(g.nodes)}
        for edge in g.get_edge_list():
            if degrees[edge[0]] < theta and degrees[edge[1]] < theta:
                degrees[edge[0]]+=1
                degrees[edge[1]]+=1
                proj_g.add_edge(edge[0], edge[1])

        self.g = proj_g

    def _sample_exp(self, param):
        return np.random.exponential(param, 1)

    def _sample_lap(self, param):
        return np.random.laplace(0, param, 1)[0]
    
    def max_degree(self):
        real = super().max_degree()
        gsf = 1
        param = gsf/self.ep2
        return round(real + self._sample_lap(param))

    def min_degree(self):
        real = super().min_degree()
        gsf = 1
        param = gsf/self.ep2
        return round(real + self._sample_lap(param))

    def avg_degree(self):
        real = super().avg_degree()
        gsf = 1
        param = gsf/self.ep2
        return real + self._sample_lap(param)

    def degree_hist(self):
        hist = super().degree_hist()
        gsf = 1
        param = gsf/self.ep2
        hist =  [round(i + self._sample_lap(param)) for i in hist]
        return hist

    def triangle_count(self):
        real = super().triangle_count()
        gsf = 2
        param = gsf/self.ep2
        return round(real +self._sample_lap(param))

    def average_edge_val(self):
        real = super().average_edge_val()
        gsf = 1
        param = gsf/self.ep2
        return real + self._sample_lap(param)
    
class Gen_Querier(Querier):
    def __init__(self, g, ep):
        super().__init__(g)
        self.ep = ep
        self.beta = 1/3
    def _sample_exp(self, param):
        return np.random.exponential(param, 1)
    
    def do_the_thing(self, D, beta, f):
        tau = math.ceil((2/self.ep) * math.log((len(D) + 1)/beta))
        ft = {}
        for j in range(2*tau):
            ft[j] = f(j)
        scores = {}
        for val in D:
            for j in range(2*tau):
                if val == ft[j]:
                    if j < tau:
                        scores[val] = self._sample_exp((self.ep/2)*(-(-tau + j - 1)))
                    elif j == tau:
                        scores[val] = self._sample_exp(0)
                    elif j > tau:
                        scores[val] = self._sample_exp(-((self.ep/2)*(tau - j)))
                    break
            if val not in scores:
                scores[val] = self._sample_exp(-((self.ep/2)*(-tau - 1)))
        
        min_score = math.inf
        min_val = None
        for val, score in scores.items():
            if score < min_score:
                min_score = score
                min_val = val
        return min_val

    def max_degree(self):
        v = super().max_degree()
        f = lambda j: abs(v - j)
        return self.do_the_thing([i for i in range(self.g.nodes)], self.beta, f)

    def min_degree(self):
        v = super().min_degree()
        f = lambda j: abs(v - j)
        return self.do_the_thing([i for i in range(self.g.nodes)], self.beta, f)

    def avg_degree(self):
        v = super().avg_degree()
        f = lambda j: abs(v - j)
        return self.do_the_thing([i for i in range(self.g.nodes)], self.beta, f)

    def degree_hist(self):
        hist = super().degree_hist()
        new_hist = []
        for i in range(len(hist)):
            f = lambda j: abs(hist[i] - j)
            new_hist.append(self.do_the_thing([i for i in range(self.g.nodes)], self.beta, f))
        return new_hist

    def triangle_count(self):
        v = super().triangle_count()
        f = lambda j: abs(v - j)
        nc3 = math.comb(self.g.nodes, 3)
        D = [i for i in range(nc3)]
        return self.do_the_thing(D, self.beta, f)

    def average_edge_val(self):
        v = super().average_edge_val()
        f = lambda j: abs(v - j)
        return self.do_the_thing([i/1000 for i in range(1000)], self.beta, f)
    
if __name__=="__main__":
    n = 100
    g = Graph(n)
    g.init_er(20/n, True)
    rq = Querier(g)
    ep = 10
    lap_q = Node_Lap_Querier(g, ep)
    elap_q = Edge_Lap_Querier(g, ep)
    g_gen_q = Graph_Gen_Querier(g, ep)
    prj_q = Proj_Querier(g, ep/2, ep/2, 200)
    gen_q = Gen_Querier(g, ep)
    queriers = [rq, lap_q, elap_q, g_gen_q, prj_q, gen_q]
    for q in queriers:
        print("===",type(q).__name__,"===")
        queries = [q.min_degree, q.max_degree, q.avg_degree, q.triangle_count, q.average_edge_val]
        for query in queries:
            print(query.__name__, query())
    g.visualize()
    prj_q.g.visualize()