import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from skdim.id import MLE # Maximum-Likelihood ID esimation


def model_param_count(model):
    return sum([np.prod(p.shape) for p in model.parameters()])


def plot_images(images, mean, std, n):
    denorm_transform = T.Normalize((-torch.tensor(mean) / torch.tensor(std)).tolist(), (1.0 / torch.tensor(std)).tolist())
    x = denorm_transform(images)
    x = torch.clip(x, 0., 1.)

    _, axs = plt.subplots(n // 10, 10, figsize=(11, 5))
    for idx, ax in enumerate(axs.flatten()):
        ax.imshow(x[idx].permute(1, 2, 0).numpy())
        ax.axis('off')
    plt.tight_layout()
    plt.show()


class IndexedDataset(Dataset):
    def __init__(self, ds):
        super().__init__()
        self.data = ds

    def __getitem__(self, idx):
        return (idx, *self.data[idx])

    def __len__(self):
        return len(self.data)


class RepresentationBuffer:
    def __init__(self, ds_size, repr_size, device='cpu'):
        self.ds_size = ds_size
        self.device = device

        self.last_epoch_repr = torch.zeros(ds_size, repr_size, requires_grad=False, device=device)

    def push(self, batch):
        idx, x = batch
        self.last_epoch_repr[idx].data = x.to(self.device).data 


class MeanSquareDistancesStat:
    def __init__(self, ds_size, n_epochs, hidden_dim, repr_buffer):
        """
        ds_size: size of dataset
        n_epochs: number of epochs during training
        hidden_dim: dimension of representations
        repr_buffer: buffer that stores last epoch representations
        """
        self.buffer = repr_buffer
        self.msd_hist = torch.zeros(n_epochs, ds_size, requires_grad=False)
                
        self.n_epochs = n_epochs

        self.cur_epoch = 0

    @torch.no_grad()
    def msd(self, x_prev, x_next):
        return torch.mean((x_next - x_prev) ** 2, dim=1)

    def push(self, batch):        
        idx, x_repr_cur = batch

        if self.cur_epoch > 0:
            x_repr_prev = self.buffer.last_epoch_repr[idx]
            self.msd_hist[self.cur_epoch, idx] = self.msd(x_repr_prev, x_repr_cur).cpu()
        
        self.buffer.push(batch)

    def inc_epoch(self):
        if self.cur_epoch == self.n_epochs: raise ValueError
        self.cur_epoch += 1

