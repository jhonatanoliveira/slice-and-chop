import numpy as np
import math
import networkx as nx
from collections import deque

LOG_ZERO = -1e3

class SPN():

    def __init__(self):
        self._curr_node_id = -1
        self.nodes = []

    def get_next_node_id(self):
        self._curr_node_id += 1
        return self._curr_node_id

    def add_node(self,node):
        self.nodes.append(node)

    def get_leaves(self):
        return [n for n in self.nodes if isinstance(n,LeafNode)]

    def get_root(self):
        poss_root = [n for n in self.nodes if len(n.predecessors) == 0]
        if len(poss_root) != 1:
            raise ValueError("SPN should contain only one root node.")
        return poss_root[0]

    def get_node_by_id(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise ValueError("Node id '{}' not found in SPN".format(node_id))

    def transform_nodes(self):
        """
        Fill in all successor nodes for each SPN node (note that during training only predecessor nodes were filled)
        And transform all nodes from IDs (integers) to Node objects
        """
        to_visit = deque(self.get_leaves())
        visited = []
        while len(to_visit) > 0:
            curr_node = to_visit.popleft()
            visited.append(curr_node)
            node_predecessors = []
            for predecessor in curr_node.predecessors:
                node_predecessor = self.get_node_by_id(predecessor)
                node_predecessor.successors.append(curr_node)
                node_predecessors.append(node_predecessor)
                if (not node_predecessor in visited) and (not node_predecessor in to_visit):
                    to_visit.append(node_predecessor)
            curr_node.predecessors = node_predecessors

    def evaluate(self,evidence):
        # Initialize leaf nodes with evidence
        for leaf in self.get_leaves():
            var_value = evidence[leaf.variables[0]]
            leaf.init_evidence(var_value)
        root = self.get_root()
        value = root.evaluate()
        return value


    def log_likelihood(self, dataset):
        log_likelihood = 0
        for i in range(dataset.shape[0]):
            evidence_row = dataset[i,:]
            value = self.evaluate(evidence_row)
            if value > 0:
                raise ValueError("Likelihood expected to be positive (probability between 0 and 1)")
            log_likelihood += value
        return log_likelihood / dataset.shape[0]


    def draw(self,file_name="spn.dot"):
        from networkx.drawing.nx_pydot import write_dot
        graph = nx.DiGraph()
        for node in self.nodes:
            node_label = "None"
            if isinstance(node,SumNode):
                node_label = "+"
            elif isinstance(node,ProductNode):
                node_label = "x"
            elif isinstance(node,LeafNode):
                node_label = "^"
            if len(node.predecessors) > 0:
                for predecessor in node.predecessors:
                    if len(node.weight)>0:
                        graph.add_edge(predecessor.id,node.id,{"weight":node.weight})
                    else:
                        graph.add_edge(predecessor.id,node.id)
                graph.node[node.id]["label"] = node_label
            else:
                graph.add_node(node.id, {"label": node_label})
        write_dot(graph, file_name)







class Node():

    def __init__(self,id,predecessors,weight,variables):
        self.id = id
        self.predecessors = predecessors
        # TODO: since for learning we don't need successors (build SPN top-down), we aren't adding successors as parameters to constructor
        # But it would be better to add both predecessors and successors as optional parameters
        self.successors = []
        self.weight = weight
        self.variables = variables

    def __hash__(self):
        return hash(self.id)

    def __eq__(self,other):
        if isinstance(other,Node):
            return self.id == other.id
        else:
            return False

    def evaluate(self):
        raise Exception("Method not implemented for this node type.")






class ProductNode(Node):

    def __init__(self,id,predecessors,weight,variables):
        # Initializations
        super().__init__(id,predecessors,weight,variables)

    def evaluate(self):
        log_value = 0
        for successor in self.successors:
            log_value += successor.evaluate()
        ### DEBUG
        # print(">>> Product node with var "+str(self.variables)+" --> "+str(log_value))
        ###---DEBUG
        return log_value






class SumNode(Node):

    def __init__(self,id,predecessors,weight,variables):
        # Initializations
        super().__init__(id,predecessors,weight,variables)

    def evaluate(self):
        pair_succ = []
        max_log = LOG_ZERO
        successor_values = []
        for successor in self.successors:
            w_sum = successor.evaluate() + math.log(successor.weight)
            if w_sum > max_log:
                max_log = w_sum
            successor_values.append(w_sum)

        sum_val = 0
        for w_sum in successor_values:
            sum_val += math.exp(w_sum - max_log)

        log_value = LOG_ZERO
        if sum_val > 0:
            log_value = math.log(sum_val) + max_log

        ### DEBUG
        # print(">>> Sum node with var "+str(self.variables)+" --> "+str(log_value))
        ###---DEBUG

        return log_value





class LeafNode(Node):

    def __init__(self,id,predecessors,weight,variables):
        # Initializations
        super().__init__(id,predecessors,weight,variables)




class CategoricalSmoothedNode(LeafNode):

    def __init__(self,id,predecessors,weight,variable,cardinality,data,alpha=0.1):

        if len(data.shape) < 2 or data.shape[1] != 1:
            raise ValueError("Categorial Smoothed Node expected a column vector of shape N x 1.")

        # Initializations
        super().__init__(id,predecessors,weight,variable)
        self.cardinality = cardinality

        # Constants
        self.alpha = alpha

        # Compute frequencies and probabilities
        self.frequencies = np.zeros(self.cardinality)
        self.compute_frequencies(data)
        self.probabilities = np.zeros(len(self.frequencies))
        self.compute_probabilities()

    def compute_frequencies(self, data):
        for row in range(data.shape[0]):
            self.frequencies[data[row,0]] += 1

    def compute_probabilities(self):
        freqs_sum = sum(self.frequencies)
        
        for i, freq in enumerate(self.frequencies):
            log_freq = LOG_ZERO
            if (freq + self.alpha) > 0:
                log_freq = math.log(freq + self.alpha)
            self.probabilities[i] = (log_freq - math.log(freqs_sum + self.cardinality * self.alpha))

    def init_evidence(self,var_value):
        self.evidence = var_value

    def evaluate(self,var_value=None):
        if not var_value:
            var_value = self.evidence
        log_value = self.probabilities[var_value]
        return log_value
