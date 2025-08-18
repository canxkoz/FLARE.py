#
import torch
from torch import nn
import torch.distributed as dist

import numpy as np

import os
import random

__all__ = [
    "set_seed",
    'set_num_threads',
    "select_device",

    'is_torchrun',
    'dist_backend',
    'dist_setup',
    'dist_finalize',

    "num_parameters",

    "mean_std",
    "normalize",
    "unnormalize",

    "r2",
    
    # versioning hell
    'to_numpy',
    'check_package_version_lteq'
]

#=======================================================================#
def to_numpy(t: torch.Tensor):
    '''
    Torch 1.10 compatible equivalent of `t.numpy(force=True)`.
    '''
    return t.detach().cpu().resolve_conj().resolve_neg().numpy()

def check_package_version_lteq(pkg: str, version: str):
    import importlib.metadata
    from packaging import version as packaging_version
    
    try:
        current_version = importlib.metadata.version(pkg)
        return packaging_version.parse(current_version) <= packaging_version.parse(version)
    except importlib.metadata.PackageNotFoundError:
        # If package not found, assume it doesn't meet the version requirement
        return False

#=======================================================================#
def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.set_float32_matmul_precision('high')
        print(f"Setting float32 matmul precision to high")
    except:
        print(f"Could not set float32 matmul precision to high.")

    return

#=======================================================================#
def set_num_threads(threads=None):
    if threads is not None:
        threads = os.cpu_count()

    torch.set_num_threads(threads)

    os.environ["OMP_NUM_THREADS"]        = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"]   = str(threads)
    os.environ["MKL_NUM_THREADS"]        = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"]    = str(threads)

    return

#=======================================================================#
def select_device(device=None, verbose=False):
    if device is not None:
        return device

    if is_torchrun():
        if not dist.is_initialized():
            dist_setup()
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        return torch.device(LOCAL_RANK)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if verbose:
        print(f'using device {device}.')

    return device

def dist_backend():
    if dist.is_nccl_available():
        return 'nccl'
    elif dist.is_gloo_available():
        return 'gloo'
    elif dist.is_mpi_available():
        return 'mpi'
    else:
        raise RuntimeError("No suitable backend found!")

def is_torchrun():
    required_env_vars = ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    return all(var in os.environ for var in required_env_vars)

def dist_setup():
    backend = dist_backend()
    if backend != 'nccl':
        print(f'using {backend} backend for torch.distributed.')

    if is_torchrun():
        GLOBAL_RANK = int(os.environ["RANK"])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(LOCAL_RANK)

        if check_package_version_lteq('torch', '2'): # not sure about version
            dist.init_process_group(
                backend,
                rank=GLOBAL_RANK,
                world_size=WORLD_SIZE,
            )
        else:
            dist.init_process_group(
                backend=backend,
                rank=GLOBAL_RANK,
                world_size=WORLD_SIZE,
                device_id=torch.device(LOCAL_RANK),
            )
    else:
        pass

    return

def dist_finalize():
    if dist.is_initialized():
        dist.destroy_process_group()
    return

#=======================================================================#
def num_parameters(model : nn.Module):
    return sum(p.numel() for p in model.parameters())

def mean_std(x: torch.tensor, channel_dim=-1):
    dims = list(range(x.ndim))
    del dims[channel_dim]
    keepdim = (channel_dim != -1) and (channel_dim != x.ndim-1)

    x_bar = x.mean(dims, keepdim=keepdim)
    x_std = x.std( dims, keepdim=keepdim)

    return x_bar, x_std

def normalize(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x - shift) / scale

def unnormalize(x_norm: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x_norm * scale + shift

#=======================================================================#

def r2(y_pred, y_true):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    y_mean = torch.mean(y_true)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-6)
    return r2.item()

#=======================================================================#
#
