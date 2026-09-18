import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

def get_connected_subgraph_nodes(A, target_size=50):
    num_nodes = A.shape[0]
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    
    for start_node in range(num_nodes):
        if visited[start_node]:
            continue
            
        q = [start_node]
        component = []
        visited[start_node] = True
        
        while q:
            curr = q.pop(0)
            component.append(curr)
            neighbors = torch.where(A[curr] > 0)[0].tolist()
            for nbr in neighbors:
                if not visited[nbr]:
                    visited[nbr] = True
                    q.append(nbr)
                    
        if len(component) >= target_size:
            return torch.tensor(component[:target_size])
    return None

class PROTEINSDataset(Dataset):
    def __init__(self, root='./data_dir', k=10):
        super().__init__()
        self.dataset = TUDataset(root=root, name='PROTEINS')
        self.k = k
        self.laplacians = []
        self.eigvals = []
        self.eigvecs = []
        
        for data in self.dataset:
            if data.num_nodes < 50:
                continue
                
            # Get dense adjacency matrix
            A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
            
            # Make sure it's undirected and binary
            A = A + A.T
            A = (A > 0).float()
            
            # Find a 50-node connected subgraph
            subgraph_nodes = get_connected_subgraph_nodes(A, target_size=50)
            if subgraph_nodes is None:
                continue
                
            # Induce subgraph
            A_sub = A[subgraph_nodes][:, subgraph_nodes]
            
            # Degree matrix
            D_sub = torch.diag(A_sub.sum(dim=1))
            
            # Laplacian
            L_sub = D_sub - A_sub
            
            # Compute eigendecomposition
            ev, evec = torch.linalg.eigh(L_sub)
            
            self.laplacians.append(L_sub)
            self.eigvals.append(ev[:k])
            self.eigvecs.append(evec[:, :k])

    def __len__(self):
        return len(self.laplacians)

    def __getitem__(self, idx):
        return self.laplacians[idx], self.eigvals[idx], self.eigvecs[idx]
