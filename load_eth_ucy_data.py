import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json

class ETHUCYDataset(Dataset):
    """Dataset class for loading pre-processed ETH/UCY data"""
    
    def __init__(self, data_dir="eth_ucy_processed", split='train', 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Args:
            data_dir: Directory containing the .npy files
            split: 'train', 'val', or 'test'
            train_ratio, val_ratio, test_ratio: Split ratios
        """
        self.data_dir = data_dir
        
        # Load metadata
        with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)
        
        print("=" * 50)
        print("LOADING ETH/UCY DATASET")
        print("=" * 50)
        print(f"Metadata: {self.metadata}")
        
        # Load data
        trajectories = np.load(os.path.join(data_dir, "trajectories.npy"))
        labels = np.load(os.path.join(data_dir, "labels.npy"))
        
        print(f"Loaded trajectories shape: {trajectories.shape}")
        print(f"Loaded labels shape: {labels.shape}")
        
        # Create splits
        num_samples = len(trajectories)
        indices = np.random.permutation(num_samples)
        
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        if split == 'train':
            self.trajectories = torch.FloatTensor(trajectories[train_indices])
            self.labels = torch.FloatTensor(labels[train_indices])
            print(f"\n✅ Train set: {len(self.trajectories)} samples")
        elif split == 'val':
            self.trajectories = torch.FloatTensor(trajectories[val_indices])
            self.labels = torch.FloatTensor(labels[val_indices])
            print(f"\n✅ Validation set: {len(self.trajectories)} samples")
        elif split == 'test':
            self.trajectories = torch.FloatTensor(trajectories[test_indices])
            self.labels = torch.FloatTensor(labels[test_indices])
            print(f"\n✅ Test set: {len(self.trajectories)} samples")
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]

# ============================================
# TEST THE LOADER
# ============================================
if __name__ == "__main__":
    # Create datasets
    train_dataset = ETHUCYDataset(split='train')
    val_dataset = ETHUCYDataset(split='val')
    test_dataset = ETHUCYDataset(split='test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("\n" + "=" * 50)
    print("TESTING DATALOADERS")
    print("=" * 50)
    
    # Test train loader
    for batch_idx, (obs, fut) in enumerate(train_loader):
        print(f"\nTrain batch {batch_idx + 1}:")
        print(f"  Observed shape: {obs.shape}")
        print(f"  Future shape: {fut.shape}")
        print(f"  Sample values:")
        print(f"    Observed first position: ({obs[0,0,0]:.2f}, {obs[0,0,1]:.2f})")
        print(f"    Future first position: ({fut[0,0,0]:.2f}, {fut[0,0,1]:.2f})")
        break
    
    print("\n" + "=" * 50)
    print("✅ DATASET READY FOR TRAINING!")
    print("=" * 50)
    print("\nUse in your model like this:")
    print("""
    for epoch in range(num_epochs):
        for batch_obs, batch_fut in train_loader:
            # batch_obs: [batch_size, 8, 2] - observed trajectory
            # batch_fut: [batch_size, 12, 2] - future trajectory to predict
            
            # Your model forward pass
            predictions = model(batch_obs)
            loss = loss_function(predictions, batch_fut)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """)