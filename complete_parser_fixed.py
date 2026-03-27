import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class ETHUCYParser:
    """Complete parser for ETH/UCY datasets"""
    
    def __init__(self, data_path, obs_len=8, pred_len=12):
        self.data_path = data_path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trajectories = []
        self.labels = []
        
    def parse_eth_file(self, filepath, scene_name):
        """Parse ETH format (8 columns)"""
        print(f"\n📄 Parsing ETH file: {os.path.basename(filepath)}")
        
        try:
            # ETH format: frame ped_id x y z? vx vy vz? (8 columns)
            data = np.loadtxt(filepath)
            print(f"   Raw shape: {data.shape}")
            
            # Extract relevant columns: frame, ped_id, x, y (columns 0,1,2,4)
            frames = data[:, 0].astype(int)
            ped_ids = data[:, 1].astype(int)
            x = data[:, 2]
            y = data[:, 4]
            
            self._process_pedestrians(frames, ped_ids, x, y, scene_name)
            print(f"   ✅ Processed {len(np.unique(ped_ids))} pedestrians")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    def parse_ucy_file(self, filepath, scene_name):
        """Parse UCY .vsp format (with headers)"""
        print(f"\n📄 Parsing UCY file: {os.path.basename(filepath)}")
        
        frames = []
        ped_ids = []
        x_coords = []
        y_coords = []
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    if ' - ' in line or len(line.split()) >= 4:
                        start_idx = i
                        break
            
            for line in lines[start_idx:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Remove trailing ' -' and split
                line = line.replace(' -', '')
                parts = line.split()
                
                if len(parts) >= 4:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        frame = int(float(parts[2]))
                        ped_id = int(float(parts[3]))
                        
                        x_coords.append(x)
                        y_coords.append(y)
                        frames.append(frame)
                        ped_ids.append(ped_id)
                    except ValueError:
                        continue
            
            if frames:
                frames = np.array(frames)
                ped_ids = np.array(ped_ids)
                x_coords = np.array(x_coords)
                y_coords = np.array(y_coords)
                
                self._process_pedestrians(frames, ped_ids, x_coords, y_coords, scene_name)
                print(f"   ✅ Processed {len(np.unique(ped_ids))} pedestrians, {len(frames)} points")
            else:
                print("   ❌ No valid data found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    def _process_pedestrians(self, frames, ped_ids, x, y, scene_name):
        """Group by pedestrian ID and create trajectories - more flexible"""
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
            ped_frames = ped_frames[sort_idx]
            
            # Create sliding windows - MORE FLEXIBLE
            seq_len = self.obs_len + self.pred_len
            if len(ped_x) >= seq_len:
                # Don't check frame continuity strictly
                for i in range(len(ped_x) - seq_len + 1):
                    obs = np.column_stack([ped_x[i:i+self.obs_len], 
                                          ped_y[i:i+self.obs_len]])
                    fut = np.column_stack([ped_x[i+self.obs_len:i+self.obs_len+self.pred_len],
                                          ped_y[i+self.obs_len:i+self.obs_len+self.pred_len]])
                    self.trajectories.append(obs)
                    self.labels.append(fut)
        
        print(f"      Scene {scene_name}: Created {len([p for p in unique_peds if len(ped_x[mask]) >= seq_len])} trajectories")
    
    def load_all(self):
        """Load all files"""
        # ETH files
        eth_files = [
            ('biwi_eth.txt', 'eth'),
            ('biwi_hotel.txt', 'hotel')
        ]
        
        for filename, scene in eth_files:
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                self.parse_eth_file(filepath, scene)
            else:
                print(f"⚠️ File not found: {filename}")
        
        # UCY files
        ucy_files = [
            ('crowds_zara01.txt', 'zara1'),
            ('crowds_zara02.txt', 'zara2'),
            ('crowds_zara03.txt', 'zara3'),
            ('students001.txt', 'students01'),
            ('students003.txt', 'students03'),
            ('uni_examples.txt', 'univ')
        ]
        
        for filename, scene in ucy_files:
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                self.parse_ucy_file(filepath, scene)
            else:
                print(f"⚠️ File not found: {filename}")
        
        print(f"\n{'='*50}")
        print(f"✅ Total trajectories loaded: {len(self.trajectories)}")
        print(f"{'='*50}")
        
        return self.trajectories, self.labels

# Main execution
if __name__ == "__main__":
    dataset_path = r"C:\Users\Jaima\Downloads\eth_ucy_final_structure"
    
    print("=" * 50)
    print("ETH/UCY COMPLETE PARSER (FIXED)")
    print("=" * 50)
    
    # Create parser
    parser = ETHUCYParser(dataset_path, obs_len=8, pred_len=12)
    
    # Load all data
    trajectories, labels = parser.load_all()
    
    if len(trajectories) > 0:
        print(f"\n✅ Success! Loaded {len(trajectories)} trajectories")
        print(f"   Trajectory shape: {trajectories[0].shape}")
        print(f"   Label shape: {labels[0].shape}")
        
        # Convert to numpy arrays
        trajectories = np.array(trajectories)
        labels = np.array(labels)
        
        print(f"\nFinal shapes:")
        print(f"   Trajectories: {trajectories.shape}")
        print(f"   Labels: {labels.shape}")
        
        # Create PyTorch dataset
        class TrajectoryDataset(Dataset):
            def __init__(self, traj, lbl):
                self.traj = torch.FloatTensor(traj)
                self.lbl = torch.FloatTensor(lbl)
            def __len__(self):
                return len(self.traj)
            def __getitem__(self, idx):
                return self.traj[idx], self.lbl[idx]
        
        import torch
        dataset = TrajectoryDataset(trajectories, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Test one batch
        print("\nTesting dataloader...")
        for batch_idx, (obs, fut) in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  Observed shape: {obs.shape}")  # [batch, 8, 2]
            print(f"  Future shape: {fut.shape}")    # [batch, 12, 2]
            
            # Print sample values
            print(f"\nSample trajectory (first in batch):")
            print(f"  Observed positions (first 3):")
            for i in range(3):
                print(f"    t+{i}: ({obs[0,i,0]:.2f}, {obs[0,i,1]:.2f})")
            print(f"  Future positions (first 3):")
            for i in range(3):
                print(f"    t+{8+i}: ({fut[0,i,0]:.2f}, {fut[0,i,1]:.2f})")
            break
    else:
        print("\n❌ No trajectories loaded.")
        
        # Debug: Show what's in the files
        print("\nDebugging file contents:")
        test_file = os.path.join(dataset_path, "crowds_zara01.txt")
        if os.path.exists(test_file):
            print(f"\nFirst 10 lines of {test_file}:")
            with open(test_file, 'r') as f:
                for i, line in enumerate(f):
                    if i < 10:
                        print(f"  Line {i+1}: {line.strip()}")
                    else:
                        break