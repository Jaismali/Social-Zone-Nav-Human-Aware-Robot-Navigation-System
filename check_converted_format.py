import numpy as np
import os

converted_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_converted"

print("=" * 50)
print("CHECKING CONVERTED FILE FORMAT")
print("=" * 50)

# Check the first few lines of a converted file
file_to_check = os.path.join(converted_path, "crowds_zara01.txt")

if os.path.exists(file_to_check):
    print(f"\n📄 Checking: {file_to_check}")
    print("-" * 30)
    
    # Try reading with numpy
    try:
        data = np.loadtxt(file_to_check)
        print(f"Data shape: {data.shape}")
        print(f"Number of columns: {data.shape[1] if len(data.shape) > 1 else 1}")
        print(f"\nFirst 5 rows:")
        for i in range(min(5, len(data))):
            print(f"  Row {i+1}: {data[i]}")
    except Exception as e:
        print(f"Error reading with numpy: {e}")
        
        # Try reading as text
        print("\nReading as text:")
        with open(file_to_check, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  Line {i+1}: {line.strip()}")
                else:
                    break
else:
    print(f"❌ File not found: {file_to_check}")

print("\n" + "=" * 50)
print("FIX: Let's create a properly formatted version")
print("=" * 50)

# Create a new folder for properly formatted data
fixed_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_fixed"
os.makedirs(fixed_path, exist_ok=True)

# Copy ETH files (they should be fine)
import shutil
eth_files = ["biwi_eth.txt", "biwi_hotel.txt"]
for f in eth_files:
    src = os.path.join(converted_path, f)
    dst = os.path.join(fixed_path, f)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"✅ Copied {f}")

# For UCY files, we need to ensure they have the right format
# Let's recreate them with the correct format
ucy_files = [
    ("crowds_zara01.txt", "crowds_zara01.txt"),
    ("crowds_zara02.txt", "crowds_zara02.txt"),
    ("crowds_zara03.txt", "crowds_zara03.txt"),
    ("students001.txt", "students001.txt"),
    ("students003.txt", "students003.txt"),
    ("uni_examples.txt", "uni_examples.txt"),
]

for src_name, dst_name in ucy_files:
    src_path = os.path.join(converted_path, src_name)
    dst_path = os.path.join(fixed_path, dst_name)
    
    if os.path.exists(src_path):
        try:
            # Read the data
            data = np.loadtxt(src_path)
            
            # Ensure it has 4 columns (frame, ped_id, x, y)
            if len(data.shape) == 2 and data.shape[1] == 4:
                # Already correct format
                shutil.copy(src_path, dst_path)
                print(f"✅ {src_name} already has correct format")
            else:
                print(f"⚠️ {src_name} has {data.shape[1]} columns, needs 4 columns")
                
                # If it has 3 columns, we need to add frame numbers
                if data.shape[1] == 3:
                    # Assume columns are [ped_id, x, y], add frame column
                    frame_col = np.arange(len(data)).reshape(-1, 1)
                    new_data = np.hstack([frame_col, data])
                    np.savetxt(dst_path, new_data, fmt=['%d', '%d', '%.6f', '%.6f'])
                    print(f"✅ Fixed {src_name} - added frame column")
                else:
                    print(f"❓ Unknown format for {src_name}")
        except Exception as e:
            print(f"❌ Error processing {src_name}: {e}")

print(f"\n✅ Fixed files saved to: {fixed_path}")