"""Batched fixed-point and linear solvers for the implicit recurrence tail.

Every solver treats dim 0 as independent systems and flattens the rest.
"""
from contextlib import nullcontext
from typing import Callable, Optional

import torch
from torch import Tensor


def _fp32(t: Tensor):
    return torch.autocast(device_type=t.device.type, enabled=False) if t.is_cuda else nullcontext()


def _rel_norm(r: Tensor, x: Tensor, eps: float = 1e-6) -> Tensor:
    return r.flatten(1).norm(dim=1) / (x.flatten(1).norm(dim=1) + eps)


def picard(f: Callable[[Tensor], Tensor], x0: Tensor, tol: float, max_steps: int, min_steps: int = 0):
    x, x_prev = x0, x0
    res = torch.full((x0.shape[0],), float("inf"), device=x0.device)
    steps = 0
    for steps in range(1, max_steps + 1):
        x_prev, x = x, f(x).to(x0.dtype)
        res = _rel_norm(x - x_prev, x)
        if steps >= min_steps and bool(res.max() < tol):
            break
    return x, x_prev, {"steps": steps, "residual": res}


def anderson(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    tol: float,
    max_steps: int,
    min_steps: int = 0,
    m: int = 5,
    lam: float = 1e-4,
    beta: float = 1.0,
):
    """Type-II Anderson acceleration (Walker & Ni) on the flattened per-sample state."""
    B = x0.shape[0]
    x0f = x0.reshape(B, -1)
    N = x0f.shape[1]
    dtype, device = x0.dtype, x0.device
    X = torch.zeros(B, m, N, dtype=dtype, device=device)
    F = torch.zeros(B, m, N, dtype=dtype, device=device)
    X[:, 0] = x0f
    F[:, 0] = f(x0).to(dtype).reshape(B, -1)
    if max_steps < 2:
        res = _rel_norm(F[:, 0] - X[:, 0], F[:, 0])
        return F[:, 0].view_as(x0), x0, {"steps": 1, "residual": res}
    X[:, 1] = F[:, 0]
    F[:, 1] = f(F[:, 0].view_as(x0)).to(dtype).reshape(B, -1)

    H = torch.zeros(B, m + 1, m + 1, dtype=torch.float32, device=device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(B, m + 1, 1, dtype=torch.float32, device=device)
    y[:, 0] = 1

    res = _rel_norm(F[:, 1] - X[:, 1], F[:, 1])
    steps = 2
    k_prev = 1
    for k in range(2, max_steps):
        n = min(k, m)
        with _fp32(x0):
            G = (F[:, :n] - X[:, :n]).float()
            H[:, 1 : n + 1, 1 : n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, device=device)
            alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[:, 1 : n + 1, 0].to(dtype)
            X[:, k % m] = beta * torch.bmm(alpha[:, None], F[:, :n])[:, 0] + (1 - beta) * torch.bmm(alpha[:, None], X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).to(dtype).reshape(B, -1)
        res = _rel_norm(F[:, k % m] - X[:, k % m], F[:, k % m])
        steps = k + 1
        k_prev = k
        if steps >= min_steps and bool(res.max() < tol):
            break
    return F[:, k_prev % m].view_as(x0), X[:, k_prev % m].view_as(x0), {"steps": steps, "residual": res}


def neumann(A: Callable[[Tensor], Tensor], b: Tensor, tol: float, max_steps: int):
    """Solves (I - J) v = b as v = sum_i J^i b, where A(y) = y - J y."""
    v, t = b, b
    steps = 0
    res = _rel_norm(t, v)
    for steps in range(1, max_steps + 1):
        t = t - A(t)
        v = v + t
        res = _rel_norm(t, v)
        if bool(res.max() < tol):
            break
    return v, {"steps": steps, "residual": res}


def gmres(A: Callable[[Tensor], Tensor], b: Tensor, tol: float, max_steps: int, restart: int = 20, x0: Optional[Tensor] = None):
    """Batched restarted GMRES(restart) with modified Gram-Schmidt Arnoldi."""
    B = b.shape[0]
    bf = b.reshape(B, -1).float()
    dtype, device = b.dtype, b.device
    x = torch.zeros_like(bf) if x0 is None else x0.reshape(B, -1).float()
    b_norm = bf.norm(dim=1) + 1e-6
    total = 0
    res = torch.full((B,), float("inf"), device=device)
    while total < max_steps:
        r = bf - A(x.to(dtype).view_as(b)).reshape(B, -1).float()
        beta = r.norm(dim=1)
        res = beta / b_norm
        if bool(res.max() < tol):
            break
        m = min(restart, max_steps - total)
        V = torch.zeros(B, m + 1, bf.shape[1], device=device)
        Hs = torch.zeros(B, m + 1, m, device=device)
        V[:, 0] = r / beta[:, None].clamp_min(1e-12)
        e1 = torch.zeros(B, m + 1, 1, device=device)
        e1[:, 0, 0] = beta
        j_used = 0
        for j in range(m):
            w = A(V[:, j].to(dtype).view_as(b)).reshape(B, -1).float()
            with _fp32(b):
                for i in range(j + 1):
                    Hs[:, i, j] = (w * V[:, i]).sum(dim=1)
                    w = w - Hs[:, i, j, None] * V[:, i]
                Hs[:, j + 1, j] = w.norm(dim=1)
                V[:, j + 1] = w / Hs[:, j + 1, j, None].clamp_min(1e-12)
                j_used = j + 1
                total += 1
                yj = torch.linalg.lstsq(Hs[:, : j + 2, : j + 1], e1[:, : j + 2]).solution
                rj = (e1[:, : j + 2] - torch.bmm(Hs[:, : j + 2, : j + 1], yj)).norm(dim=(1, 2))
            if bool((rj / b_norm).max() < tol) or total >= max_steps:
                break
        with _fp32(b):
            yj = torch.linalg.lstsq(Hs[:, : j_used + 1, :j_used], e1[:, : j_used + 1]).solution
            x = x + torch.bmm(yj.transpose(1, 2), V[:, :j_used])[:, 0]
    r = bf - A(x.to(dtype).view_as(b)).reshape(B, -1).float()
    res = r.norm(dim=1) / b_norm
    return x.to(dtype).view_as(b), {"steps": total, "residual": res}


def anderson_linear(A: Callable[[Tensor], Tensor], b: Tensor, tol: float, max_steps: int, m: int = 5, lam: float = 1e-4):
    """Anderson on the fixed-point form v = b + J v of (I - J) v = b."""
    v, _, info = anderson(lambda y: b + (y - A(y)), b, tol, max_steps, m=m, lam=lam)
    return v, info


FIXED_POINT_SOLVERS = {"picard": picard, "anderson": anderson}
LINEAR_SOLVERS = {"neumann": neumann, "gmres": gmres, "anderson": anderson_linear}
