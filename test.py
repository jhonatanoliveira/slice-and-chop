import numpy as np
from learnspn import LearnSPN
from spn import CategoricalSmoothedNode

ts_filename = "data/nltcs.ts.data"

D = np.loadtxt(ts_filename, delimiter=",", dtype=np.uint32)
learner = LearnSPN(D)

c1,c2 = learner.chop(np.arange(0,D.shape[0]),np.arange(0,D.shape[1]))
c = learner.slice(np.arange(0,D.shape[0]),np.arange(0,D.shape[1]))

n = CategoricalSmoothedNode(0,[],0,2,D[[1,2,3,4,5],:][:,[3]])
print(n.frequencies)
print(n.probabilities)