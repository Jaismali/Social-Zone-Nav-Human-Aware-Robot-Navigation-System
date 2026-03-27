import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# Path to your dataset
dataset_path = r"C:\Users\Jaima\Downloads\eth_ucy_final_structure"

print("=" * 50)
print("MANUALLY LOADING ETH/UCY DATASET")
print("=" * 50)

class ETHUCYDataset(Dataset):
    """Simple dataset for ETH/UCY without trajdata"""
    
    def __init__(self, data_path, obs_len=8, pred_len=12):
        self.data_path = data_path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trajectories = []
        self.labels = []
        
        # Load all files
        files = {
            'eth': 'biwi_eth.txt',
            'hotel': 'biwi_hotel.txt',
            'zara1': 'crowds_zara01.txt',
            'zara2': 'crowds_zara02.txt',
            'zara3': 'crowds_zara03.txt',
            'students': ['students001.txt', 'students003.txt'],
            'univ': 'uni_examples.txt'
        }
        
        for scene_name, filename in files.items():
            if isinstance(filename, list):
                for f in filename:
                    self._load_file(os.path.join(data_path, f), scene_name)
            else:
                self._load_file(os.path.join(data_path, filename), scene_name)
        
        print(f"\nLoaded {len(self.trajectories)} trajectories")
    
    def _load_file(self, filepath, scene_name):
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            return
            
        print(f"\n📄 Loading {os.path.basename(filepath)}...")
        
        # Try different ways to read the file
        try:
            # Try reading with numpy
            data = np.loadtxt(filepath)
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {data.shape[1] if len(data.shape) > 1 else 1}")
            
            if len(data.shape) == 2 and data.shape[1] == 4:
                # Assuming [frame, ped_id, x, y]
                frames = data[:, 0]
                ped_ids = data[:, 1]
                x = data[:, 2]
                y = data[:, 3]
                
                # Group by pedestrian
                unique_peds = np.unique(ped_ids)
                for ped in unique_peds:
                    mask = ped_ids == ped
                    ped_frames = frames[mask]
                    ped_x = x[mask]
                    ped_y = y[mask]
                    
                    # Sort by frame
                    sort_idx = np.argsort(ped_frames)
                    ped_x = ped_x[sort_idx]
                    ped_y = ped_y[sort_idx]
                    
                    # Create trajectory if long enough
                    if len(ped_x) >= self.obs_len + self.pred_len:
                        for i in range(len(ped_x) - self.obs_len - self.pred_len + 1):
                            obs = np.column_stack([ped_x[i:i+self.obs_len], 
                                                  ped_y[i:i+self.obs_len]])
                            fut = np.column_stack([ped_x[i+self.obs_len:i+self.obs_len+self.pred_len],
                                                  ped_y[i+self.obs_len:i+self.obs_len+self.pred_len]])
                            self.trajectories.append(obs)
                            self.labels.append(fut)
                            
                print(f"   Extracted {len(unique_peds)} pedestrians from {scene_name}")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]

# Create dataset
print("\n" + "=" * 50)
print("CREATING DATASET")
print("=" * 50)

dataset = ETHUCYDataset(dataset_path)

if len(dataset) > 0:
    print(f"\n✅ Success! Dataset has {len(dataset)} trajectories")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test one batch
    print("\nTesting dataloader...")
    for batch_idx, (obs, fut) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Observed shape: {obs.shape}")  # [batch, obs_len, 2]
        print(f"  Future shape: {fut.shape}")    # [batch, pred_len, 2]
        break
else:
    print("\n❌ No trajectories loaded. Check file formats.")
    
    # Let's peek at the raw files
    print("\nPeeking at raw files:")
    test_files = ['biwi_eth.txt', 'crowds_zara01.txt']
    for f in test_files:
        filepath = os.path.join(dataset_path, f)
        if os.path.exists(filepath):
            print(f"\n📄 First 5 lines of {f}:")
            with open(filepath, 'r') as fp:
                for i, line in enumerate(fp):
                    if i < 5:
                        print(f"  Line {i+1}: {line.strip()}")
                    else:
                        break