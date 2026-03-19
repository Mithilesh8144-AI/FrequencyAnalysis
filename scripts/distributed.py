"""Distributed training and AMP utilities shared across training scripts."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup_distributed():
    """Initialize DDP if launched with torchrun, otherwise return single-GPU config.

    Returns:
        rank: Global rank (0 for single-GPU).
        local_rank: Local GPU index.
        world_size: Total number of processes.
        device: torch.device for this rank.
        is_distributed: Whether DDP is active.
    """
    is_distributed = "RANK" in os.environ
    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device, is_distributed


def cleanup_distributed():
    """Clean up DDP process group if active."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model, device, local_rank, is_distributed, find_unused_parameters=False):
    """Move model to device and wrap in DDP if distributed.

    Returns the (possibly wrapped) model. Use unwrap_model() to access
    the underlying module for saving state_dict / calling custom methods.
    """
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=find_unused_parameters)
    return model


def unwrap_model(model):
    """Get the underlying module from a DDP-wrapped model (or return as-is)."""
    return model.module if isinstance(model, DDP) else model


def make_dataloader(dataset, batch_size, shuffle, is_distributed, **kwargs):
    """Create a DataLoader with DistributedSampler when using DDP.

    Returns (dataloader, sampler). Caller must call sampler.set_epoch(epoch)
    each epoch when distributed to ensure proper shuffling.
    """
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # sampler handles shuffling
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, **kwargs)
    return loader, sampler


def is_main(rank):
    """Only rank 0 should print / save."""
    return rank == 0
