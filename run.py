import numpy as np
import networkx as nx
import pandas as pd
import itertools as it
import scipy.stats as stats


from math import exp, log, log10
from collections import deque

from sklearn.cluster import KMeans
from sklearn.neighbors.kde import KernelDensity



P_VALUE_CUT = 0.0015 # product nodes
N_CLUSTERS = 3 # sum nodes
GAUSSIAN_BANDWIDTH = 0.2 # leaf nodes



class NodeIdGenerator():
    def __init__(self):
        self.id = -1
    def get_next(self):
        self.id += 1
        return self.id



def g_test(D, col1, col2):
    data = pd.DataFrame(D)
    contingency = pd.crosstab(data[col1], data[col2])
    chi2, p_value, degree_freedom, expected_freq = stats.chi2_contingency(contingency, lambda_="log-likelihood")
    return p_value



def select(T,V,D):
    row_idx = np.array(T)
    col_idx = np.array(V)
    D_i = D[row_idx[:, None], col_idx]
    return D_i


def index_groups(groups, elements, non_empty=True):
    ind_groups = [ [] for _ in range(len(groups)) ]
    for elem_pos, group in enumerate(groups):
        ind_groups[group].append(elements[elem_pos])
    if (non_empty):
        ind_groups = [ig for ig in ind_groups if len(ig)>0]
    return ind_groups



def get_independent_subsets(T,V,D):
    # Initially, all variables are in diff groups
    groups = list(range(len(V)))
    for v1, v2 in it.combinations(range(len(V)),2):
        p_val = g_test(select(T,V,D), v1, v2)
        # Test for dependency: low values of p-value means low confidency in the 
        # null-hypotesis (which says that the variables are independent)
        if p_val < P_VALUE_CUT:
            groups[v2] = groups[v1]
    V_is = index_groups(groups,V)
    return V_is



def get_clusters(T,V,D):
    wrap_n_clusters = N_CLUSTERS
    if len(T) < N_CLUSTERS:
        wrap_n_clusters = len(T)
    kmeans = KMeans(n_clusters=wrap_n_clusters, random_state=0).fit(select(T,V,D))
    T_is = index_groups(kmeans.labels_,T)
    return T_is



def learn_spn(T,V,D,G, build_leaf=False):
    if build_leaf or len(V) == 1:
        kde = KernelDensity(kernel='gaussian', bandwidth=GAUSSIAN_BANDWIDTH).fit(select(T,V,D))
        leaf_node = G.node_id_gen.get_next()
        G.add_node(leaf_node, {"type": "leaf", "label": V, "distribution": kde, "variables": V})
        return leaf_node
    else:
        V_is = get_independent_subsets(T,V,D)
        chop_success = len(V_is) > 1
        if chop_success:
            product_node = G.node_id_gen.get_next()
            G.add_node(product_node, {"type":"product","label":"x"})
            for i, V_i in enumerate(V_is):
                child = learn_spn(T,V_i,D,G)
                G.add_edge(product_node, child)
            return product_node
        else:
            T_is = get_clusters(T,V,D)
            slice_success = not ((len(T_is) == 1) and (len(T_is[0]) == len(T)))
            if slice_success:
                sum_node = G.node_id_gen.get_next()
                G.add_node(sum_node, {"type":"sum", "label": "+"})
                for T_i in T_is:
                    child = learn_spn(T_i,V,D,G)
                    weight = len(T_i)/len(T)
                    G.add_edge(sum_node, child, {"weight": weight, "label": "{:.4f}".format(weight)})
                return sum_node
            else:
                return learn_spn(T_is[0],V,D,G,build_leaf=True)



def evaluate(d,G):
    vr = {n: 0 for n in G.nodes()}
    to_visit = deque([x for x in G.nodes_iter() if G.out_degree(x)==0])
    
    while len(to_visit) > 0:
        node = to_visit.popleft()
        n_data = G.node[node]
        if n_data["type"] == "leaf":
            leaf_var_values = d[n_data["variables"]]
            vr[node] = exp(n_data["distribution"].score_samples(leaf_var_values))
        elif n_data["type"] == "product":
            prod = 1
            for child in G.successors(node):
                prod = prod * vr[child]
            vr[node] = prod
        elif n_data["type"] == "sum":
            t_sum = 0
            for child in G.successors(node):
                t_sum = t_sum + vr[child]
            vr[node] = t_sum
        to_visit.extend(G.predecessors(node))

    root = [x for x in G.nodes_iter() if G.out_degree(x)==0][0]
    return vr[root]



if __name__ == "__main__":
    ts_filename = "data/plants.test.data"

    D = np.loadtxt(ts_filename, delimiter=",")
    D = D[0:200,:]

    V = list(range(0,D.shape[1]))
    T = list(range(0,D.shape[0]))

    G = nx.DiGraph()
    G.node_id_gen = NodeIdGenerator()

    print(">>> Learning SPN")
    learn_spn(T,V,D,G)

    np.random.seed(seed=134)
    d = np.random.randint(2, size=len(V)).reshape(-1,1)
    print(">>> Querying SPN")
    p = evaluate(d, G)
    print(p)

    # import matplotlib.pyplot as plt
    # nx.draw_networkx_edge_labels(G)
    # plt.show()

    from networkx.drawing.nx_pydot import write_dot
    write_dot(G, "t.dot")
