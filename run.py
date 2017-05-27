import numpy as np
import networkx as nx
import pandas as pd
import itertools as it
import scipy.stats as stats


from math import exp, log
from collections import deque

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture



P_VALUE_CUT = 0.0015 # product nodes
N_CLUSTERS = 3 # sum nodes
N_COMPONENTS = 3 # leaf nodes



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
        # Test for dependency: low values of p-value means independent
        if p_val > P_VALUE_CUT:
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
        wrap_n_components = N_COMPONENTS
        if len(V) < N_COMPONENTS:
            wrap_n_components = len(V)
        gm = GaussianMixture(n_components=wrap_n_components, covariance_type='full').fit(select(T,V,D))
        leaf_node = G.node_id_gen.get_next()
        G.add_node(leaf_node, {"type": "leaf", "label": V, "distribution": gm, "variables": V})
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
    if len(d.shape) > 1 and d.shape[0] != 1:
        raise ValueError("Only one instance for evaluation.")
    root = [x for x in G.nodes_iter() if G.in_degree(x)==0]
    if len(root) > 1:
        raise ValueError("SPN with more than one root.")

    root = root[0]
    vr = {n: 0 for n in G.nodes()}
    to_visit = deque([x for x in G.nodes_iter() if G.out_degree(x)==0])
    
    last_value_node = 0
    while len(to_visit) > 0:
        node = to_visit.popleft()
        n_data = G.node[node]
        if n_data["type"] == "leaf":
            instance = select([0],n_data["variables"],d)
            log_value = n_data["distribution"].score_samples(instance)
            if log_value > 0:
                log_value = 0
            vr[node] = exp(log_value)
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
        else:
            raise ValueError("Node type '{}' not supported.".format(n_data["type"]))
        to_visit.extend([n for n in G.predecessors(node) if not n in to_visit])
        last_value_node = vr[node]
    return vr[root]



def log_likelihood(D,G):
    ll = 0
    for row in range(D.shape[0]):
        d = D[row,:].reshape(1,D.shape[1])
        p = evaluate(d, G)
        ll += log(p)
    return ll



def normalize(G):
    sum_nodes = [s for s,d in G.nodes(True) if d["type"]=="sum"]
    for s_node in sum_nodes:
        norm = 0
        s_children = G.successors(s_node)
        for child in s_children:
            norm += G.edge[s_node][child]["weight"]
        for child in s_children:
            new_weight = G.edge[s_node][child]["weight"] / norm
            G.edge[s_node][child]["weight"] = new_weight
            G.edge[s_node][child]["label"] = "{:.4f}".format(new_weight)



if __name__ == "__main__":
    ts_filename = "data/plants.ts.data"

    D = np.loadtxt(ts_filename, delimiter=",")
    D = D[0:30,:]

    V = list(range(0,D.shape[1]))
    T = list(range(0,D.shape[0]))

    G = nx.DiGraph()
    G.node_id_gen = NodeIdGenerator()

    from time import time
    t_0 = time()
    print(">>> Learning SPN")
    learn_spn(T,V,D,G)
    print(">>> Exec time: "+str(time()-t_0))

    t_0 = time()
    print(">>> Normalizing SPN")
    normalize(G)
    print(">>> Exec time: "+str(time()-t_0))

    # np.random.seed(0)
    # d1 = np.random.randint(2,size=(1,len(V)))
    # print(d1)
    # p1 = evaluate(d1, G)
    # print(">>> Final Probability: ")
    # print(p1)
    # d2 = np.random.randint(2,size=(1,len(V)))
    # print(d2)
    # p2 = evaluate(d2, G)
    # print(p2)

    t_0 = time()
    print(">>> Compute Log-Likelihood")
    ll = log_likelihood(D,G)
    print(ll)
    print(">>> Exec time: "+str(time()-t_0))

    # import matplotlib.pyplot as plt
    # nx.draw_networkx_edge_labels(G)
    # plt.show()

    from networkx.drawing.nx_pydot import write_dot
    write_dot(G, "t.dot")



# - When few instances (determine a minimum acceptable) but still with some features:
#    - Break into a product node with children as follows (of course, each one with one feature):
#       - Create a Laplace smoothed by counting the frequency of values: log_freq = log(freq + alpha),  freqs[i] = (log_freq - log(freqs_sum + vals * alpha))
#       - Basically: how many times each value appeared in this column
