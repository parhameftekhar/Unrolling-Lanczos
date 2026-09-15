import random
from torch.utils.data import DataLoader, Subset
from .sbm import SBMDataset

def get_dataloaders(dataset_name, args):
    if dataset_name.lower() == 'sbm':
        dataset = SBMDataset(
            n_graphs=args.n_graphs, 
            n=args.n, 
            k_blocks=args.k_blocks, 
            p_in=args.p_in, 
            p_out=args.p_out, 
            k=args.k, 
            base_seed=args.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split dataset
    n_graphs = len(dataset)
    indices = list(range(n_graphs))
    random.Random(args.seed).shuffle(indices)
    
    train_frac, val_frac = 0.6, 0.2
    n_train = int(n_graphs * train_frac)
    n_val = int(n_graphs * val_frac)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader
