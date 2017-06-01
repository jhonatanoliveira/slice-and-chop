import math
from collections import deque
import numpy as np
from sklearn.mixture import GaussianMixture

from spn import SPN
from spn import ProductNode, SumNode, CategoricalSmoothedNode

class LearnSPN():

    def __init__(self, dataset, g_factor=0.0015, n_clusters=2, leaf_alpha=0.1, min_instances=100, random_seed=0):
        # Learning constants
        self.g_factor = g_factor
        self.n_clusters = n_clusters
        self.rand_gen = np.random.RandomState(random_seed)
        self.alpha = leaf_alpha
        self.min_instances = min_instances
        # Dataset information
        self.dataset = dataset
        self.variables = np.arange(1,self.dataset.shape[1],dtype=np.uint32)
        self.cardinalities = self.get_feature_cardinalities()



    def chop(self, rows, columns):
        """
        Get two independent subsets of features.
        Randomly select one feature and from it check dependency with all other ones.
        As soon as a dependency is found, add the dependent feature to subset and start looking for another dependent feature from the new one.
        """
        n_features = len(columns)

        features_consideration = np.ones(n_features, dtype=bool)
        rand_feature = self.rand_gen.randint(0, n_features)
        features_consideration[rand_feature] = False

        dependent_features = np.zeros(n_features, dtype=bool)
        dependent_features[rand_feature] = True

        features_to_process = deque()
        features_to_process.append(rand_feature)

        while features_to_process:
            current_feature = features_to_process.popleft()
            feature1 = columns[current_feature]

            # features to remove later
            features_to_remove = np.zeros(n_features, dtype=bool)

            for other_feature in features_consideration.nonzero()[0]:
                feature2 = columns[other_feature]
                if not self.g_test(feature1,feature2,rows):
                    features_to_remove[other_feature] = True
                    dependent_features[other_feature] = True
                    features_to_process.append(other_feature)

            # now removing from future considerations
            features_consideration[features_to_remove] = False

        # translating remaining features
        first_component = columns[dependent_features]
        second_component = columns[~dependent_features]

        return first_component, second_component


    def slice(self,rows,columns):
        """
        Get two similar components from all instances (rows).
        """
        def index_groups(groups, elements, non_empty=True):
            ind_groups = [ [] for _ in range(len(groups)) ]
            for elem_pos, group in enumerate(groups):
                ind_groups[group].append(elements[elem_pos])
            if (non_empty):
                ind_groups = [ig for ig in ind_groups if len(ig)>0]
            return ind_groups
        data_chunk = self.dataset[rows,:][:,columns]
        wrap_n_clusters = min(self.n_clusters,len(rows)-1)
        gmm = GaussianMixture(n_components=wrap_n_clusters,random_state=self.rand_gen).fit(data_chunk)
        clustering = gmm.predict(data_chunk)
        return index_groups(clustering,rows)



    def train(self):

        spn = SPN()

        data_chunks_to_process = deque()

        all_rows = np.arange(0,self.dataset.shape[0])
        all_columns = np.arange(0,self.dataset.shape[1])

        root_node_id = spn.get_next_node_id()
        root_node_predecessors = []
        data_chunks_to_process.append((root_node_id,root_node_predecessors,np.array([]),all_rows,all_columns))

        # TODO: not sure why start by clustering instances instead of finding independent features
        first_run = True
        while (len(data_chunks_to_process) > 0):
            curr_node_id, curr_node_predecessors, weight, rows_chunk, columns_chunk = data_chunks_to_process.popleft()
            if len(columns_chunk) == 1:
                # Laplace smoothed leaf node
                variable = columns_chunk[0]
                chunked_data = self.dataset[rows_chunk,:][:,columns_chunk]
                leaf_node = CategoricalSmoothedNode(curr_node_id,curr_node_predecessors,weight,np.array([variable]),self.cardinalities[variable],chunked_data,alpha=self.alpha)
                spn.add_node(leaf_node)
            elif (len(rows_chunk) < self.min_instances) and (len(columns_chunk) > 1):
                # Factorize into a product node with all features being leaf nodes
                children = []
                for column in columns_chunk:
                    child_id = spn.get_next_node_id()
                    children.append(child_id)
                    data_chunks_to_process.append((child_id,[curr_node_id],np.array([]),rows_chunk,np.array([column])))
                product_node = ProductNode(curr_node_id,curr_node_predecessors,weight,columns_chunk)
                spn.add_node(product_node)
            else:
                # In first run, cluster instances
                chop_features = False
                if first_run:
                    first_run = False
                else:
                    first_chop_chunk, second_chop_chunk = self.chop(rows_chunk,columns_chunk)
                    if (len(first_chop_chunk) > 0) and (len(second_chop_chunk) > 0):
                        chop_features = True
                # Chop features or slice instances
                if chop_features:
                    child_id_1 = spn.get_next_node_id()
                    data_chunks_to_process.append((child_id_1,[curr_node_id],np.array([]),rows_chunk,first_chop_chunk))
                    child_id_2 = spn.get_next_node_id()
                    data_chunks_to_process.append((child_id_2,[curr_node_id],np.array([]),rows_chunk,second_chop_chunk))

                    product_node = ProductNode(curr_node_id,curr_node_predecessors,weight,columns_chunk)
                    spn.add_node(product_node)
                else:
                    first_slice_chunk, second_slice_chunk = self.slice(rows_chunk,columns_chunk)

                    n_instances = len(rows_chunk)
                    child_id_1 = spn.get_next_node_id()
                    weight_child_1 = len(first_slice_chunk) / n_instances
                    data_chunks_to_process.append((child_id_1,[curr_node_id],np.array([weight_child_1]),first_slice_chunk,columns_chunk))
                    child_id_2 = spn.get_next_node_id()
                    weight_child_2 = len(second_slice_chunk) / n_instances
                    data_chunks_to_process.append((child_id_2,[curr_node_id],np.array([weight_child_2]),second_slice_chunk,columns_chunk))

                    sum_node = SumNode(curr_node_id,curr_node_predecessors,weight,columns_chunk)
                    spn.add_node(sum_node)
        # Fill in all successor nodes for each SPN node (note that during training only predecessor nodes were filled)
        # And transform all nodes from IDs to Node objects
        spn.transform_nodes()
        return spn



    # Utilities
    # ---------
    def get_feature_cardinalities(self):
        cardinalities = np.zeros(self.dataset.shape[1],dtype=np.uint32)
        for column in range(self.dataset.shape[1]):
            values = np.unique(self.dataset[:,column])
            # TODO: better solution for handling case where variable does not change
            # TOFIX: Solution: compute cardinality before learning. The problem is that by computing cardinality later in the slice and chop part, a chunk of the data might not contain all possible values anymore, thus wrongly saying that the cardinality is 1.
            cardinality = len(values)
            if cardinality < 2:
                cardinality = 2
            cardinalities[column] = cardinality
        return cardinalities



    def g_test(self, feature1, feature2, rows):
        # TODO: not sure why need to maintain order of features
        if feature1 > feature2:
            tmp_feat = feature1
            feature1 = feature2
            feature2 = tmp_feat
        
        size_feature1 = self.cardinalities[feature1]
        size_feature2 = self.cardinalities[feature2]

        frequencies_feat1 = np.zeros(size_feature1,dtype=np.uint32)
        frequencies_feat2 = np.zeros(size_feature2,dtype=np.uint32)
        frequencies_combined = np.zeros([size_feature1,size_feature2],dtype=np.uint32)

        for row in rows:
            frequencies_combined[self.dataset[row,feature1],self.dataset[row,feature2]] += 1

        for row in range(size_feature1):
            for column in range(size_feature2):
                count = frequencies_combined[row,column]
                frequencies_feat1[row] += count
                frequencies_feat2[column] += count

        frequencies_feat1_non_zero = np.count_nonzero(frequencies_feat1)
        frequencies_feat2_non_zero = np.count_nonzero(frequencies_feat2)

        degree_of_freedom = (frequencies_feat1_non_zero - 1) * (frequencies_feat2_non_zero - 1)

        g_value = 0.0
        n_instances = len(rows)
        for i, freq_feat1 in enumerate(frequencies_feat1):
            for j, freq_feat2 in enumerate(frequencies_feat2):
                count = frequencies_combined[i, j]
                if count != 0:
                    exp_count = freq_feat1 * freq_feat2 / n_instances
                    g_value += count * math.log(count / exp_count)
        g_value *= 2
        p_value = 2 * degree_of_freedom * self.g_factor + 0.001

        return g_value < p_value
