import torch

def subspace_error(Q, T, U_k, k):
    """E²_sub = (1/k) · ‖(I − Û_k Û_k^T) U_k‖²_F"""
    B = T.shape[0]
    _, U_T = torch.linalg.eigh(T)
    k_use = min(k, U_T.shape[-1])
    U_hat = torch.bmm(Q, U_T[:, :, :k_use])
    U_hat, _ = torch.linalg.qr(U_hat)
    U_k_use = U_k[:, :, :k_use]
    inner = torch.bmm(U_hat.transpose(-1, -2), U_k_use)
    proj = U_k_use - torch.bmm(U_hat, inner)
    return proj.pow(2).sum() / (B * k_use)

def compute_reg_loss(alpha_vec, beta_vec, reg_alpha, reg_beta, device):
    """L = λ_α · mean(α²) − λ_β · mean(β²)."""
    loss_alpha = alpha_vec.pow(2).mean()
    loss_beta = torch.tensor(0.0, device=device)
    if beta_vec.numel() > 0:
        loss_beta = beta_vec.pow(2).mean()
    return reg_alpha * loss_alpha - reg_beta * loss_beta, loss_alpha, loss_beta
