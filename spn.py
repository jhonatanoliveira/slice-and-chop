import numpy as np
import math

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






class Node():

    def __init__(self,id,predecessors):
        self.id = id
        self.predecessors = predecessors






class CategoricalSmoothedNode(Node):

    def __init__(self,id,predecessors,variable,cardinality,data,alpha=0.1):

        if len(data.shape) < 2 or data.shape[1] != 1:
            raise ValueError("Categorial Smoothed Node expected a column vector of shape N x 1.")

        # Initializations
        super().__init__(id,predecessors)
        self.variable = variable
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

    def evaluate(self,var_value):
        return self.probabilities[var_value]





 class ProductNode(Node):

    def __init__(self,id,predecessors,variables):
        # Initializations
        super().__init__(id,predecessors)
        self.variables = variables

