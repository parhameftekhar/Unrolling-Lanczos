import torch
from torch.utils.data import Dataset

def make_sbm_laplacian(n=100, k_blocks=4, p_in=0.3, p_out=0.02, seed=42):
    """Combinatorial Laplacian of a Stochastic Block Model graph."""
    torch.manual_seed(seed)
    block_size = n // k_blocks
    labels = torch.arange(n) // block_size
    probs = torch.where(
        labels.unsqueeze(0) == labels.unsqueeze(1),
        torch.full((n, n), p_in),
        torch.full((n, n), p_out),
    )
    A = torch.bernoulli(probs).triu(diagonal=1)
    A = A + A.T
    D = torch.diag(A.sum(dim=1))
    return D - A

class SBMDataset(Dataset):
    def __init__(self, n_graphs=50, n=100, k_blocks=4, p_in=0.3, p_out=0.02, k=10, base_seed=0):
        super().__init__()
        self.laplacians = []
        self.eigvals = []
        self.eigvecs = []
        
        for i in range(n_graphs):
            L = make_sbm_laplacian(n=n, k_blocks=k_blocks, p_in=p_in,
                                   p_out=p_out, seed=base_seed + i)
            ev, evec = torch.linalg.eigh(L)
            self.laplacians.append(L)
            self.eigvals.append(ev[:k])
            self.eigvecs.append(evec[:, :k])

    def __len__(self):
        return len(self.laplacians)

    def __getitem__(self, idx):
        return self.laplacians[idx], self.eigvals[idx], self.eigvecs[idx]
