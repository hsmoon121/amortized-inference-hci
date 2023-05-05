import numpy as np
import torch


def sort_and_pad_traj_data(stat_data, traj_data, value=0):
    """
    Sort and pad trajectory data based on their lengths.

    stat_data (ndarray): static data with a shape (num_data, stat_feature_dim).
    traj_data (list): List of trajectory data, each item should have a shape (traj_length, traj_feature_dim).
    value (int, optional): Padding value, default is 0.
    ---
    outputs (tuple): Tuple containing sorted static data (torch.Tensor), sorted padded trajectory data (torch.Tensor),
               padding lengths (torch.Tensor), and sorted indices (torch.Tensor).
    """
    # Get trajectory lengths
    traj_lens = [traj.shape[0] for traj in traj_data]
    max_len = max(traj_lens)

    # Pad trajectory data
    padded_data = []
    for traj in traj_data:
        padded_data.append(np.pad(
            traj,
            ((0, max_len), (0, 0)),
            "constant",
            constant_values=value
        )[:max_len])

    # Sort trajectory data based on length
    lens = torch.LongTensor(traj_lens)
    lens, sorted_idx = lens.sort(descending=True)
    padded = max_len - lens

    # Get sorted static and trajectory data
    sorted_trajs = torch.FloatTensor(np.array(padded_data))[sorted_idx]
    sorted_stats = torch.FloatTensor(np.array(stat_data))[sorted_idx]
    return sorted_stats, sorted_trajs, padded, sorted_idx


def mask_and_pad_traj_data(traj_data, value=0):
    """
    Create a mask and pad trajectory data based on their lengths.

    traj_data (list): List of trajectory data, each item should have a shape (traj_length, traj_feature_dim).
    value (int, optional): Padding value, default is 0.
    ---
    outputs (tuple): Tuple containing padded trajectory data (torch.Tensor) and mask (torch.BoolTensor).
    """
    # Get trajectory lengths
    traj_lens = [traj.shape[0] for traj in traj_data]
    max_len = max(traj_lens)
    mask = torch.zeros((len(traj_data), max_len))

    # Pad trajectory data and create mask
    padded_data = []
    for i, traj in enumerate(traj_data):
        padded_data.append(np.pad(
            traj,
            ((0, max_len), (0, 0)),
            "constant",
            constant_values=value
        )[:max_len])
        mask[i, :traj_lens[i]] = 1

    # Get padded data and mask to tensors
    padded_trajs = torch.FloatTensor(np.array(padded_data))
    return padded_trajs, mask.bool()


def fourier_encode(x, max_freq, num_bands = 4):
    """
    Fourier feature postiion encodings
    reference: https://github.com/lucidrains/perceiver-pytorch
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[((None,) * (len(x.shape) - 1) + (...,))]

    x = x * scales * np.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


def get_auto_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)