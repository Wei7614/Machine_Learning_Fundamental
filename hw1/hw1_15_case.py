import numpy as np
from Perceptron import Perceptron

pla = Perceptron()

# load data
data_source = np.loadtxt("./hw1_15_train.dat.txt")
dataset = data_source[:, :4]
target = data_source[:, 4]
w = pla.get_weight(dataset, target, is_random_seed=False)
print(w)
