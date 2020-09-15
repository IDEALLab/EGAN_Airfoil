import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from interpolation import interpolate

class UIUCAirfoilDataset(Dataset):
    r"""UIUC Airfoil Dataset. 

    Args:
        data_fname: Directory to the UIUC airfoil .npy file.
        N: Number of data points.
        k: Degree of spline.
        D: Shifting constant. The higher the more uniform the data points are.
    Shape:
        Output: `(N, D, DP)` where D is the dimension of each point and DP is the number of data points.
    """

    def __init__(self, data_fname, N=192, k=3, D=20):
        super().__init__()
        self.airfoils = np.load(data_fname).transpose((0, 2, 1))
        if (N, k, D) == (192, 3, 20):
            self.N = N; self.k = k; self.D = D
        else:
            self.refresh(N, k, D)

    def refresh(self, N, k, D):
        self.N = N; self.k = k; self.D = D
        self.airfoils = np.array([interpolate(airfoil, N, k, D) for airfoil in self.airfoils])
    
    def __getitem__(self, index):
        return self.airfoils[index]
    
    def __len__(self):
        return len(self.airfoils)
    
    def __str__(self):
        return '<UIUC Airfoil Dataset (size={}, resolution={}, spline degree={}, uniformity={})>'.format(
            self.__len__(), self.N, self.k, self.D
        )

class NoiseGenerator:
    def __init__(self, batch: int, sizes: list=[4, 10], noise_type: list=['u', 'n']):
        super().__init__()
        self.batch = batch
        self.sizes = sizes
        self.noise_type = noise_type
        
    def __call__(self):
        noises = []
        for size, n_type in zip(self.sizes, self.noise_type):
            if n_type == 'u':
                noises.append(torch.rand(self.batch, size))
            elif n_type == 'n':
                noises.append(torch.randn(self.batch, size))
        return torch.hstack(noises)

if __name__ == '__main__':
    data_fname = './data/airfoil_interp.npy'
    dataset = UIUCAirfoilDataset(data_fname)
    print(dataset)
    dataloader = DataLoader(dataset, 128)
    print(list(dataloader)[0].shape)
