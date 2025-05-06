from Querier import *
import matplotlib.pyplot as plt

""" with open('facebook_combined.txt', 'r') as fb:
    edges = fb.read().split("\n")
    g = Graph(4039)
    for edge in edges:
        edge = edge.split(" ")
        g.add_edge(int(edge[0]), int(edge[1])) """
n = 200
p = [i / 10 for i in range(1, 10)]
ep = 5
errs = {i : [] for i in range(5)}
for pct in p:
    g = Graph(n)
    g.init_er(pct, True)
    real = Querier(g)
    n_lap = Node_Lap_Querier(g, ep)
    e_lap = Edge_Lap_Querier(g, ep)
    gph_q = Graph_Gen_Querier(g, ep)
    pj_q = Proj_Querier(g, ep/2, ep/2, 200)
    g_q = Gen_Querier(g, ep)
    queriers = [n_lap, e_lap, gph_q, pj_q, g_q]
    actual = real.degree_hist()
    for i in range(len(queriers)):
        err = 0
        for j in range(3):
            """
            measured = queriers[i].average_edge_val()
            err += abs(actual - measured)/(actual if actual != 0 else 1)
            print(queriers[i], actual, measured, err)
            """         
            measured = queriers[i].degree_hist()
            err += sum([abs(actual[k] - measured[k])/(actual[k] if actual[k] != 0 else 1) for k in range(len(actual))])/len(actual)

        err /= 3
        errs[i].append(err)

qs = ["Laplace (Node)", "Laplace (Edge)", "Graph Generation", "Projection", "General Monotonic"]
q2s = ["Max-Degree", "Min-Degree", "Average Degree", "Degree Histogram", "Triangle Count", "Average Edge Weight"]
plt.figure(dpi=150)
for q in range(len(qs)):
    plt.plot(p, errs[q], label=qs[q])
plt.xlabel("Probability of edge addition (p)")
plt.ylabel("Error %")
plt.title(f"Mechanism Performance on {q2s[3]} Query vs Connectedness (Epsilon=5, n=200)")
plt.legend()
plt.show()

""" nodes = [10, 20, 50, 100, 200, 500]
pcts = [i/10 for i in range(1, 10)]
eps = [i for i in range(1, 11)]
graphs = []
for n in nodes:
    for p in pcts:
        g = Graph(n)
        g.init_er(p, True)
        graphs.append(g)

eps = [i for i in range(1, 11)]
with open('facebook_combined.txt', 'r') as fb:
    edges = fb.read().split("\n")
    fbg = Graph(4039)
    for edge in edges:
        edge = edge.split(" ")
        fbg.add_edge(int(edge[0]), int(edge[1]))
    a_q = Querier(fbg)
    actual = [a_q.max_degree(), a_q.min_degree(), a_q.avg_degree(), a_q.triangle_count()]
    for ep in eps:
        n_lap = Node_Lap_Querier(fbg, ep)
        e_lap = Edge_Lap_Querier(fbg, ep)
        gph_q = Graph_Gen_Querier(fbg, ep)
        pj_q = Proj_Querier(fbg, ep/2, ep/2, 200)
        g_q = Gen_Querier(fbg, ep)
        queriers = [n_lap, e_lap, gph_q, pj_q, g_q]
        errs = []
        for q in queriers:
            queries = [q.max_degree, q.min_degree, q.avg_degree, q.triangle_count]
            q_errs = []
            for i in range(len(queries)):
                v = queries[i]()
                err = abs(actual[i] - v)/[actual[i] if actual[i] != 0 else 1][0]
                q_errs.append(err)
            errs.append(q_errs)
    
    with open("fb_res.txt", 'w') as fbo:
        fbo.write(str(errs))

results = []
for g in graphs:
    a_q = Querier(g)
    actual = [a_q.max_degree(), a_q.min_degree(), a_q.avg_degree(), a_q.triangle_count()]
    for ep in eps:
        q_errs = []
        n_lap = Node_Lap_Querier(g, ep)
        e_lap = Edge_Lap_Querier(g, ep)
        gph_q = Graph_Gen_Querier(g, ep)
        pj_q = Proj_Querier(g, ep/2, ep/2, 200)
        g_q = Gen_Querier(g, ep)
        queriers = [n_lap, e_lap, gph_q, pj_q, g_q]
        for q in queriers:
            queries = [q.max_degree, q.min_degree, q.avg_degree, q.triangle_count]
            errs = []
            for i in range(len(queries)):
                res = [0, 0, math.inf]
                for j in range(10):
                    v = queries[i]()/1.0
                    err = abs(actual[i] - v)/[actual[i] if actual[i] != 0 else 1][0]
                    if err > res[0]:
                        res[0] = err
                    if err < res[2]:
                        res[2] = err
                    res[1] += err
                res[1] = res[1]/10
                errs.append(res)
            q_errs.append(errs)
        results.append(q_errs)

with open('out.txt', 'w') as op:
    op.write(str(results))
 """
