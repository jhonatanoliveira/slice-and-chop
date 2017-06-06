import numpy as np
from learnspn import LearnSPN
from spn import CategoricalSmoothedNode
from time import time

random_seed = 12345

datasets = ["nltcs"]
dataset_path = "data/"

chop_methods = ["gtest", "mi"]
slice_methods = ["gmm", "kmeans"]

g_factors = [5,10,15,20]
mi_factors = [0.1,0.05,0.01,0.005]
leaf_alpha = 0.1
min_instances = [10,50,100,500]


def write_file_experiment(file, dataset, train_dataset, test_dataset, chop_method, slice_method, g_factor, mi_factor, leaf_alpha, min_instance, random_seed):

    print("*** Learning: {},{},{},{},{},{}".format(str(dataset),str(g_factor),str(leaf_alpha),str(min_instance),str(chop_method),str(slice_method)))

    learner = LearnSPN(train_dataset, chop_method=chop_method, slice_method=slice_method, g_factor=g_factor, mi_factor=mi_factor, leaf_alpha=leaf_alpha, min_instances=min_instance, random_seed=random_seed)
    t0_learning = time()
    spn = learner.train()
    t1_learning = time()

    print(">>> Computing Log-Likelihoods")
    t0_ll_train = time()
    ll_train = spn.log_likelihood(train_dataset)
    t1_ll_train = time()
    print(">>> Train LL: {}. Computed in {}".format(str(ll_train),str(t1_ll_train-t0_ll_train)))
    t0_ll_test = time()
    ll_test = spn.log_likelihood(test_dataset)
    t1_ll_test = time()
    print(">>> Computed in {} (s)".format(str(t1_ll_train-t0_ll_train)))
    print(">>> Test LL: {}. Computed in {}".format(str(ll_test),str(t1_ll_test-t0_ll_test)))

    file.write("{},{},{},{},{},{},{},{},{},{},{}".format(str(dataset),str(g_factor),str(leaf_alpha),str(min_instance),str(chop_method),str(slice_method),str(ll_train),str(ll_test),str(t1_learning-t0_learning),str(t1_ll_train-t1_ll_train),str(t1_ll_test-t1_ll_test)))




with open("experiment_results.txt","w") as file:

    for dataset in datasets:

        train_path = dataset_path+dataset+".ts.data"
        test_path = dataset_path+dataset+".test.data"

        train_dataset = np.loadtxt(train_path, delimiter=",", dtype=np.uint32)
        test_dataset = np.loadtxt(test_path, delimiter=",", dtype=np.uint32)

        for min_instance in min_instances:

            for chop_method in chop_methods:
                for slice_method in slice_methods:

                    if chop_method == "gtest":
                        for g_factor in g_factors:
                            write_file_experiment(file, dataset, train_dataset, test_dataset, chop_method, slice_method, g_factor, None, leaf_alpha, min_instance, random_seed)
                    elif chop_method == "mi":
                        for mi_factor in mi_factors:
                            write_file_experiment(file, dataset, train_dataset, test_dataset, chop_method, slice_method, None, mi_factor, leaf_alpha, min_instance, random_seed)

                                
    file.close()
