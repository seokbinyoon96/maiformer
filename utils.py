import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_model_layout(traj):
    """(B, T, N, n_vars) -> (B, T, N * n_vars), the layout MAIFormer takes."""
    B, T, N, F = traj.shape
    return traj.reshape(B, T, N * F)


def to_agent_layout(traj, n_vars=3):
    """(B, T, N * n_vars) -> (B, T, N, n_vars)."""
    B, T, L = traj.shape
    return traj.reshape(B, T, L // n_vars, n_vars)


def masked_mse(pred, target, agent_mask=None, n_vars=3):
    """MSE over real agents only.
    pred, target : (B, T, N * n_vars)
    agent_mask   : (B, N) bool, True = real agent (optional)
    """
    err = (to_agent_layout(pred, n_vars) - to_agent_layout(target, n_vars)) ** 2   # (B, T, N, F)
    if agent_mask is None:
        return err.mean()
    m = agent_mask[:, None, :, None].to(err.dtype)
    return (err * m).sum() / (m.sum() * err.size(1) * n_vars).clamp_min(1)


@torch.no_grad()
def ade_fde(pred, target, agent_mask=None, n_vars=3, dims=(0, 1)):
    """Average / final displacement error over the variates in `dims`
    (default lat, lon), in the units of pred / target (de-normalize first).
    pred, target : (B, T, N * n_vars)
    agent_mask   : (B, N) bool, True = real agent (optional)
    """
    p = to_agent_layout(pred, n_vars)[..., list(dims)]
    t = to_agent_layout(target, n_vars)[..., list(dims)]
    dist = (p - t).norm(dim=-1)                                     # (B, T, N)
    if agent_mask is None:
        return dist.mean().item(), dist[:, -1].mean().item()
    m = agent_mask.to(dist.dtype)                                   # (B, N)
    ade = (dist * m[:, None]).sum() / (m.sum() * dist.size(1)).clamp_min(1)
    fde = (dist[:, -1] * m).sum() / m.sum().clamp_min(1)
    return ade.item(), fde.item()
