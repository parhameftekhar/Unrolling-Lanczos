"""
Synthetic experiment: learnable Lanczos initialisation via inverse-polynomial CG.

Solves  (I + Σ a_t L^t) r = z,  z ~ N(0, I)  using conjugate gradient
with *learnable* step sizes α_k and momentum terms β_k.
Then v_1 = r / ||r|| is used as the Lanczos starting vector.

Evaluation: E²_sub (subspace error vs ground-truth eigenvectors).
"""

import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ──────────────────────────────────────────────────────────────────
# 0. Lanczos filter (self-contained, standard γ = 1)
# ──────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────
# 1. Graph construction & dataset
# ──────────────────────────────────────────────────────────────────
from data import get_dataloaders


# ──────────────────────────────────────────────────────────────────
# 2. Evaluation metrics
# ──────────────────────────────────────────────────────────────────

def subspace_error(Q, T, U_k, k):
    """E²_sub = (1/k) · ‖(I − Û_k Û_k^T) U_k‖²_F"""
    T_sq = T.squeeze(0)
    _, U_T = torch.linalg.eigh(T_sq)
    k_use = min(k, U_T.shape[-1])
    U_hat = Q.squeeze(0) @ U_T[:, :k_use]
    U_hat, _ = torch.linalg.qr(U_hat)
    U_k_use = U_k[:, :k_use]
    proj = U_k_use - U_hat @ (U_hat.T @ U_k_use)
    return proj.pow(2).sum() / k_use


# ──────────────────────────────────────────────────────────────────
# 3. CG Inverse-Polynomial Initialiser
# ──────────────────────────────────────────────────────────────────

class CGPolyInitializer(nn.Module):
    """
    Inverse-polynomial initialisation via learnable Conjugate Gradient.

    Solves  (I + Σ_{t=1}^T a_t L^t) r = z,   z ~ N(0, I)
    using CG with learnable step sizes α_k and momentum β_k.
    Returns v_1 = r / ||r||.

    Learnable parameters:
        • a_t   : polynomial coefficients (positive via softplus)
        • α_k   : CG step sizes per iteration
        • β_k   : CG momentum terms per iteration
    """

    def __init__(self, poly_degree: int = 3, cg_steps: int = 10, inv_poly_power: int = 1, plain_cg: bool = False):
        super().__init__()
        self.poly_degree = poly_degree
        self.cg_steps = cg_steps
        self.inv_poly_power = inv_poly_power
        self.plain_cg = plain_cg

        # a_t > 0 ensured by softplus, now distinct per inverse power
        self._coeff_raw = nn.Parameter(torch.zeros(inv_poly_power, poly_degree))

        # Learnable CG hyper-parameters, distinct per inverse power
        # We store the raw values and apply softplus in forward() to ensure > 0
        # Init α small to avoid divergence: softplus(-6.0) ≈ 0.0025
        self._cg_alpha_raw = nn.Parameter(torch.full((inv_poly_power, cg_steps), -6.0))
        # Init β near zero: softplus(-6.0) ≈ 0.0025
        self._cg_beta_raw = nn.Parameter(torch.full((inv_poly_power, cg_steps), -6.0))

    def _poly_matvec(self, L_norm, x, coeff):
        """Compute A·x = (I + Σ a_t L_norm^t) · x"""
        result = x  # I·x
        L_pow_x = x
        for t in range(self.poly_degree):
            L_pow_x = torch.bmm(L_norm, L_pow_x)
            result = result + coeff[t] * L_pow_x
        return result

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        B, N, _ = L.shape
        device, dtype = L.device, L.dtype

        # Normalise L by Gershgorin spectral-radius estimate so eigenvalues ∈ [0, ~1]
        row_sums = L.abs().sum(dim=-1).max(dim=-1).values  # (B,)
        scale = row_sums.clamp(min=1.0).view(B, 1, 1)
        L_norm = L / scale

        z = torch.randn(B, N, 1, device=device, dtype=dtype)

        # Ensure positivity
        alpha = F.softplus(self._cg_alpha_raw)
        beta = F.softplus(self._cg_beta_raw)
        coeff = F.softplus(self._coeff_raw)

        # Sequential solves for power P: x_p = A^{-1} x_{p-1}
        rhs = z
        for p in range(self.inv_poly_power):
            x = torch.zeros(B, N, 1, device=device, dtype=dtype)
            r = rhs.clone()
            p_cg = r.clone()

            if self.plain_cg:
                # Standard Conjugate Gradient without learnable alpha/beta
                r_sq = (r * r).sum(dim=1, keepdim=True)
                for i in range(self.cg_steps):
                    Ap = self._poly_matvec(L_norm, p_cg, coeff[p])
                    pAp = (p_cg * Ap).sum(dim=1, keepdim=True)
                    # avoid div by zero
                    alpha_cg = r_sq / pAp.clamp(min=1e-8)
                    x = x + alpha_cg * p_cg
                    r_new = r - alpha_cg * Ap
                    r_sq_new = (r_new * r_new).sum(dim=1, keepdim=True)
                    beta_cg = r_sq_new / r_sq.clamp(min=1e-8)
                    p_cg = r_new + beta_cg * p_cg
                    r = r_new
                    r_sq = r_sq_new
            else:
                # Learnable Conjugate Gradient
                for i in range(self.cg_steps):
                    Ap = self._poly_matvec(L_norm, p_cg, coeff[p])
                    x = x + alpha[p, i] * p_cg
                    r = r - alpha[p, i] * Ap
                    p_cg = r + beta[p, i] * p_cg
            
            rhs = x  # Output of this solve is RHS for the next

        return x / (x.norm(dim=1, keepdim=True) + 1e-8)



class LearnableLanczosInit(nn.Module):
    """Wraps LanczosFilter with a CGPolyInitializer."""

    def __init__(self, k: int, poly_degree: int = 3, cg_steps: int = 10, inv_poly_power: int = 1, plain_cg: bool = False):
        super().__init__()
        self.lanczos = LanczosFilter(k=k)
        self.v1_init = CGPolyInitializer(poly_degree=poly_degree,
                                         cg_steps=cg_steps,
                                         inv_poly_power=inv_poly_power,
                                         plain_cg=plain_cg)

    def forward(self, L):
        v_init = self.v1_init(L)
        return self.lanczos(L, v_init=v_init)


# ──────────────────────────────────────────────────────────────────
# 4. Loss & evaluation helpers
# ──────────────────────────────────────────────────────────────────

def compute_reg_loss(alpha_vec, beta_vec, reg_alpha, reg_beta, device):
    """L = λ_α · mean(α²) − λ_β · mean(β²)."""
    loss_alpha = alpha_vec.pow(2).mean()
    loss_beta = torch.tensor(0.0, device=device)
    if beta_vec.numel() > 0:
        loss_beta = beta_vec.pow(2).mean()
    return reg_alpha * loss_alpha - reg_beta * loss_beta, loss_alpha, loss_beta


@torch.no_grad()
def evaluate_loader(model, loader, k, device, n_samples=5):
    """Average E²_sub over multiple z samples for stability."""
    model.eval()
    total = 0.0
    count = 0
    for L, _, eigvec in loader:
        L = L.to(device)
        U_k = eigvec[0].to(device)
        sample_sum = 0.0
        for _ in range(n_samples):
            Q, T, _, _ = model(L)
            sample_sum += subspace_error(Q, T, U_k, k).item()
        total += sample_sum / n_samples
        count += 1
    return total / max(count, 1)


# ──────────────────────────────────────────────────────────────────
# 5. Training loop
# ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Generating SBM graphs ...")
    train_loader, val_loader, test_loader = get_dataloaders('sbm', args)
    print(f"Split: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test")

    k = args.k
    n_lanczos = 2 * k

    model = LearnableLanczosInit(
        k=n_lanczos, poly_degree=args.poly_degree, 
        cg_steps=args.cg_steps, inv_poly_power=args.inv_poly_power,
        plain_cg=args.plain_cg
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Learnable params: {n_params}  "
          f"(poly_coeff={args.poly_degree}, "
          f"cg_alpha={0 if args.plain_cg else args.cg_steps}, "
          f"cg_beta={0 if args.plain_cg else args.cg_steps}, "
          f"inv_power={args.inv_poly_power})\n")

    # --- Baseline ---
    default_sub = evaluate_loader(model, test_loader, k, device)
    print(f"Default (untrained) test E²_sub = {default_sub:.6f}\n")

    use_subspace_loss = args.loss_mode == "subspace"
    if use_subspace_loss:
        print(f"{'Epoch':>6} | {'Loss':>10} | {'trn_E²sub':>10} | {'val_E²sub':>10}")
        print("-" * 50)
    else:
        print(f"{'Epoch':>6} | {'Loss':>10} | {'Reg(α)':>10} | {'Reg(β)':>10} | "
              f"{'trn_E²sub':>10} | {'val_E²sub':>10}")
        print("-" * 75)

    best_val_sub = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_la, epoch_lb = 0.0, 0.0, 0.0

        for L, _, eigvec in train_loader:
            optimizer.zero_grad()
            L = L.to(device)
            Q, T, alpha_vec, beta_vec = model(L)

            if use_subspace_loss:
                loss = subspace_error(Q, T, eigvec[0].to(device), k)
                la, lb = torch.tensor(0.0), torch.tensor(0.0)
            else:
                loss, la, lb = compute_reg_loss(
                    alpha_vec, beta_vec, args.reg_alpha, args.reg_beta, device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_la += la.item()
            epoch_lb += lb.item()

        n_train = max(len(train_loader.dataset), 1)
        epoch_loss /= n_train
        epoch_la /= n_train
        epoch_lb /= n_train

        trn_sub = evaluate_loader(model, train_loader, k, device)
        val_sub = evaluate_loader(model, val_loader, k, device)

        if val_sub < best_val_sub:
            best_val_sub = val_sub
            best_epoch = epoch
            best_state = {k_: v.detach().cpu().clone()
                          for k_, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            if use_subspace_loss:
                print(f"{epoch:>6} | {epoch_loss:>10.6f} | "
                      f"{trn_sub:>10.6f} | {val_sub:>10.6f}")
            else:
                print(f"{epoch:>6} | {epoch_loss:>10.6f} | {epoch_la:>10.6f} | "
                      f"{epoch_lb:>10.6f} | {trn_sub:>10.6f} | {val_sub:>10.6f}")

    # --- Best model → test ---
    print(f"\nRestoring best model from epoch {best_epoch} "
          f"(val E²_sub = {best_val_sub:.6f})")
    model.load_state_dict(best_state)

    test_sub = evaluate_loader(model, test_loader, k, device)

    print("\n" + "=" * 60)
    print("Final test-set comparison")
    print("=" * 60)
    print(f"  {'Default (untrained)':>25}  E²_sub = {default_sub:.6f}")
    print(f"  {'Learned (best epoch)':>25}  E²_sub = {test_sub:.6f}")
    improvement = (default_sub - test_sub) / max(default_sub, 1e-10) * 100
    print(f"  {'Improvement':>25}         = {improvement:.2f}%")

    # Print learned params
    init = model.v1_init
    coeff = F.softplus(init._coeff_raw).data.tolist()
    alpha = F.softplus(init._cg_alpha_raw).data.tolist()
    beta = F.softplus(init._cg_beta_raw).data.tolist()
    for p in range(args.inv_poly_power):
        print(f"\n  [Power {p+1}] Poly coeffs a_t = {[f'{c:.4f}' for c in coeff[p]]}")
        if not args.plain_cg:
            print(f"  [Power {p+1}] CG step α_k     = {[f'{a:.4f}' for a in alpha[p]]}")
            print(f"  [Power {p+1}] CG momentum β_k = {[f'{b:.4f}' for b in beta[p]]}")

    g1 = torch.sigmoid(model.lanczos._gamma1_raw).data.tolist()
    g2 = (1.0 + F.softplus(model.lanczos._gamma2_raw)).data.tolist()

    return default_sub, test_sub, improvement, g1, g2


# ──────────────────────────────────────────────────────────────────
# 6. CLI
# ──────────────────────────────────────────────────────────────────

def parse_args(args=None):
    p = argparse.ArgumentParser(
        description="Train inverse-polynomial CG Lanczos init on SBM graphs.")

    # Graph
    p.add_argument("--n-graphs", type=int, default=50)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--k-blocks", type=int, default=4)
    p.add_argument("--p-in", type=float, default=0.3)
    p.add_argument("--p-out", type=float, default=0.02)

    # Lanczos & initialiser
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--poly-degree", type=int, default=3)
    p.add_argument("--cg-steps", type=int, default=10)
    p.add_argument("--inv-poly-power", type=int, default=1, help="Number of times to sequentially apply the inverse polynomial filter")
    p.add_argument("--plain-cg", action="store_true", help="Use standard CG without learnable step sizes and momentum")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--loss-mode", default="reg", choices=["reg", "subspace"])
    p.add_argument("--reg-alpha", type=float, default=0.01)
    p.add_argument("--reg-beta", type=float, default=0.01)

    # Misc
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(args)


if __name__ == "__main__":
    train(parse_args())