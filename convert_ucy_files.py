import os
import numpy as np

def convert_vsp_to_trajdata(input_file, output_file):
    """Convert UCY .vsp format to trajdata expected format"""
    
    print(f"Converting: {input_file}")
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (first two lines)
    if len(lines) > 2:
        # First line: number of splines
        # Second line: number of control points
        data_lines = lines[2:]
        
        trajectories = []
        
        for line in data_lines:
            if line.strip() and '-' in line:
                # Parse line like: "279.000000 -123.000000 0 87.397438 - (2D point, m_id)"
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        frame = float(parts[2])
                        ped_id = float(parts[3])
                        # Format: frame ped_id x y (like ETH format)
                        trajectories.append([frame, ped_id, x, y])
                    except:
                        continue
        
        # Save in ETH-like format (frame, ped_id, x, y)
        if trajectories:
            np.savetxt(output_file, np.array(trajectories), 
                      fmt=['%d', '%d', '%.6f', '%.6f'],
                      delimiter='\t')
            print(f"✅ Converted {len(trajectories)} points to {output_file}")
            return True
    
    print(f"❌ Failed to convert {input_file}")
    return False

# Paths
dataset_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_final"
output_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_converted"

# Create output folder
os.makedirs(output_path, exist_ok=True)

print("=" * 50)
print("CONVERTING UCY FILES")
print("=" * 50)

# Copy ETH files first (they're already in correct format)
import shutil
shutil.copy(os.path.join(dataset_path, "biwi_eth.txt"), 
            os.path.join(output_path, "biwi_eth.txt"))
shutil.copy(os.path.join(dataset_path, "biwi_hotel.txt"), 
            os.path.join(output_path, "biwi_hotel.txt"))
print("✅ Copied ETH files")

# Convert UCY files
ucy_files = [
    ("crowds_zara01.txt", "crowds_zara01.txt"),
    ("crowds_zara02.txt", "crowds_zara02.txt"),
    ("crowds_zara03.txt", "crowds_zara03.txt"),
    ("students001.txt", "students001.txt"),
    ("students003.txt", "students003.txt"),
    ("uni_examples.txt", "uni_examples.txt"),
    ("arxiepiskopi.txt", "arxiepiskopi.txt")
]

for input_name, output_name in ucy_files:
    input_path = os.path.join(dataset_path, input_name)
    output_path_file = os.path.join(output_path, output_name)
    
    if os.path.exists(input_path):
        convert_vsp_to_trajdata(input_path, output_path_file)
    else:
        print(f"❌ File not found: {input_name}")

print("\n" + "=" * 50)
print(f"✅ Converted files saved to: {output_path}")
print("=" * 50)