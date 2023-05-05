import numpy as np
import torch

def hist_kld(a, b, nbins=50):
    range = (min(np.min(a), np.min(b)), max(np.max(a), np.max(b)))
    ahist = np.histogram(a, bins=nbins, range=range, density=True)[0]
    bhist = np.histogram(b, bins=nbins, range=range, density=True)[0]
    return kl(ahist, bhist)

def kl(p, q):
    epsilon = 1e-5
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def gaussian_kernel_matrix(x, y, sigmas=None):
    if sigmas is None:
        sigmas = torch.Tensor([
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ])
    beta = 1. / (2. * (torch.unsqueeze(sigmas, 1)))
    dist = torch.norm(torch.unsqueeze(torch.Tensor(x), 2) - torch.Tensor(y).T, dim=1).T
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))
    kernel = torch.reshape(torch.sum(torch.exp(-s), 0), dist.shape)
    return kernel

def mmd_kernel(x, y):
    kernel = gaussian_kernel_matrix
    m, n = x.shape[0], y.shape[0]
    loss = (1.0 / (m * (m + 1))) * torch.sum(kernel(x, x))  
    loss += (1.0 / (n * (n + 1))) * torch.sum(kernel(y, y))  
    loss -= (2.0 / (m * n)) * torch.sum(kernel(x, y))
    return loss