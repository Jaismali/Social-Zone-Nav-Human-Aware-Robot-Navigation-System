import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class ETHUCYProcessor:
    """Process ETH/UCY dataset and save to disk"""
    
    def __init__(self, data_path, obs_len=8, pred_len=12):
        self.data_path = data_path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trajectories = []
        self.labels = []
        
    def parse_eth_file(self, filepath, scene_name):
        """Parse ETH format (8 columns)"""
        print(f"📄 Parsing ETH file: {os.path.basename(filepath)}")
        
        try:
            data = np.loadtxt(filepath)
            print(f"   Raw shape: {data.shape}")
            
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
        print(f"📄 Parsing UCY file: {os.path.basename(filepath)}")
        
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
        """Group by pedestrian ID and create trajectories"""
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
            
            # Create sliding windows
            seq_len = self.obs_len + self.pred_len
            if len(ped_x) >= seq_len:
                for i in range(len(ped_x) - seq_len + 1):
                    obs = np.column_stack([ped_x[i:i+self.obs_len], 
                                          ped_y[i:i+self.obs_len]])
                    fut = np.column_stack([ped_x[i+self.obs_len:i+self.obs_len+self.pred_len],
                                          ped_y[i+self.obs_len:i+self.obs_len+self.pred_len]])
                    self.trajectories.append(obs)
                    self.labels.append(fut)
    
    def process_all(self):
        """Process all dataset files"""
        print("=" * 50)
        print("PROCESSING ETH/UCY DATASET")
        print("=" * 50)
        
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
        
        # Convert to numpy arrays
        self.trajectories = np.array(self.trajectories)
        self.labels = np.array(self.labels)
        
        print(f"\n{'='*50}")
        print(f"✅ PROCESSING COMPLETE!")
        print(f"   Trajectories shape: {self.trajectories.shape}")
        print(f"   Labels shape: {self.labels.shape}")
        print(f"   Total samples: {len(self.trajectories)}")
        print(f"{'='*50}")
        
        return self.trajectories, self.labels
    
    def save_to_disk(self, save_dir="eth_ucy_processed"):
        """Save processed data to disk"""
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as numpy files
        traj_path = os.path.join(save_dir, "trajectories.npy")
        labels_path = os.path.join(save_dir, "labels.npy")
        
        np.save(traj_path, self.trajectories)
        np.save(labels_path, self.labels)
        
        # Save metadata
        metadata = {
            'num_trajectories': len(self.trajectories),
            'obs_len': self.obs_len,
            'pred_len': self.pred_len,
            'trajectory_shape': self.trajectories.shape,
            'labels_shape': self.labels.shape
        }
        
        import json
        with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✅ Data saved to '{save_dir}' folder:")
        print(f"   - trajectories.npy")
        print(f"   - labels.npy")
        print(f"   - metadata.json")
        
        return save_dir

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # Path to your raw dataset files
    dataset_path = r"C:\Users\Jaima\Downloads\eth_ucy_final_structure"
    
    # Create processor
    processor = ETHUCYProcessor(dataset_path, obs_len=8, pred_len=12)
    
    # Process all data
    trajectories, labels = processor.process_all()
    
    # Save to disk
    save_folder = processor.save_to_disk("eth_ucy_processed")
    
    print(f"\n🎉 Dataset ready! Saved to: {os.path.abspath(save_folder)}")