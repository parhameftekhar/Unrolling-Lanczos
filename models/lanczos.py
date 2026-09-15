import torch
import torch.nn as nn
import torch.nn.functional as F

class LanczosFilter(nn.Module):
    """Standard k-step Lanczos decomposition with learnable γ."""

    def __init__(self, k: int, eps: float = 1e-4):
        super().__init__()
        self.k = k
        self.eps = eps
        self._gamma1_raw = nn.Parameter(torch.full((k,), 7.0))
        self._gamma2_raw = nn.Parameter(torch.full((k,), -7.0))

    def forward(self, L, v_init=None):
        B, N, _ = L.shape
        device, dtype = L.device, L.dtype
        eps = self.eps
        k = min(self.k, N)

        gamma1 = torch.sigmoid(self._gamma1_raw)
        gamma2 = 1.0 + F.softplus(self._gamma2_raw)

        if v_init is not None:
            v = v_init.to(dtype=dtype)
            v = v / v.norm(dim=1, keepdim=True).clamp(min=eps)
        else:
            v = torch.full((B, N, 1), 1.0 / (N ** 0.5), device=device, dtype=dtype)

        Q_cols, alphas, betas = [], [], []

        w = torch.bmm(L, v)
        alpha = (v * w).sum(dim=1, keepdim=True)
        w = w - gamma1[0] * alpha * v
        alphas.append(alpha.squeeze(-1))
        Q_cols.append(v)

        for j in range(1, k):
            beta_norm = w.norm(dim=1, keepdim=True)
            if beta_norm.max().item() < eps:
                break
            v_new = w / beta_norm.clamp(min=eps)
            w = torch.bmm(L, v_new)
            alpha = (v_new * w).sum(dim=1, keepdim=True)
            w = w - gamma1[j] * alpha * v_new - gamma2[j] * beta_norm * v
            betas.append(beta_norm.squeeze(-1))
            alphas.append(alpha.squeeze(-1))
            Q_cols.append(v_new)
            v = v_new

        k_actual = len(Q_cols)
        Q = torch.cat(Q_cols, dim=-1)
        alpha_vec = torch.cat(alphas, dim=-1)
        T = torch.diag_embed(gamma1[:k_actual] * alpha_vec)
        if betas:
            beta_vec = torch.cat(betas, dim=-1)
            idx = torch.arange(k_actual - 1, device=device)
            T[:, idx, idx + 1] = gamma2[1:k_actual] * beta_vec
            T[:, idx + 1, idx] = beta_vec
        else:
            beta_vec = torch.zeros(B, 0, device=device, dtype=dtype)

        return Q, T, alpha_vec, beta_vec
