import numpy as np
from learnspn import LearnSPN
from spn import CategoricalSmoothedNode
from time import time

ts_filename = "data/nltcs.ts.data"

D = np.loadtxt(ts_filename, delimiter=",", dtype=np.uint32)
learner = LearnSPN(D)

# c1,c2 = learner.chop(np.arange(0,D.shape[0]),np.arange(0,D.shape[1]))
# c = learner.slice(np.arange(0,D.shape[0]),np.arange(0,D.shape[1]))

# n = CategoricalSmoothedNode(0,[],0,2,D[[1,2,3,4,5],:][:,[3]])

tic = time()
spn = learner.train()
tac = time()
print(">>> Structure learned in {}".format(tac-tic))

tic = time()
v =  spn.log_likelihood(D)
tac = time()
print(">>> Log Likelihood: "+str(v)+" - in "+str(tac-tic))
