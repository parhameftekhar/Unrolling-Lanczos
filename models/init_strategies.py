import torch
import torch.nn as nn
import torch.nn.functional as F

from .lanczos import LanczosFilter

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
