import numpy as np

data_fname = './data/airfoil_interp.npy'
X = np.load(data_fname)

print(X.shape)