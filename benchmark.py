from ising_sampling import IsingModel
import numpy as np
from time import time


# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput

# with PyCallGraph(output=GraphvizOutput()):
numspin = 30
n = 1000

np.random.seed(56426481)
h = np.random.normal(size=numspin)
j = np.random.normal(size=(numspin, numspin))
j += j.T
j[np.diag_indices_from(j)] = np.zeros(numspin)
# np.random.seed()

ntrials = 20

t = time()
for _ in range(10):
    model = IsingModel(numspin)
    model.import_ising01(h, j)

    for x in model.sample(n):
        pass
t = time() - t
print('Time taken, ' + str(ntrials) + ' trials:')
print(str(t) + ' seconds.')
