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


from models import LearnableLanczosInit

# ──────────────────────────────────────────────────────────────────
# 1. Graph construction & dataset
# ──────────────────────────────────────────────────────────────────
from data import get_dataloaders


from metrics import subspace_error, compute_reg_loss

# ──────────────────────────────────────────────────────────────────
# 4. Loss & evaluation helpers
# ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_loader(model, loader, k, device, n_samples=5):
    """Average E²_sub over multiple z samples for stability."""
    model.eval()
    total = 0.0
    count = 0
    for L, _, eigvec in loader:
        L = L.to(device)
        U_k = eigvec.to(device)
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

    print(f"Loading {args.dataset} dataset ...")
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, args)
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
                loss = subspace_error(Q, T, eigvec.to(device), k)
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

    # Dataset
    p.add_argument("--dataset", type=str, default="sbm", choices=["sbm", "proteins"], help="Dataset to use")

    # Graph (SBM specific)
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--loss-mode", default="reg", choices=["reg", "subspace"])
    p.add_argument("--reg-alpha", type=float, default=0.01)
    p.add_argument("--reg-beta", type=float, default=0.01)

    # Misc
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(args)


if __name__ == "__main__":
    train(parse_args())