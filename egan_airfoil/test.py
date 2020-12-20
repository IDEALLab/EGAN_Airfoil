from utils.dataloader import UIUCAirfoilDataset
import numpy as np

data_fname = '../data/train.npy'
X_train = np.load(data_fname)

data = UIUCAirfoilDataset(X_train, 128, 4, 20)

print(data[0].shape)