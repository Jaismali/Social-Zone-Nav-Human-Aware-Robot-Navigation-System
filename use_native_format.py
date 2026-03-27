from trajdata import UnifiedDataset
import os
import shutil

# Create a folder with the exact structure trajdata expects
native_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_native"
os.makedirs(native_path, exist_ok=True)

print("=" * 50)
print("SETTING UP NATIVE UCY FORMAT")
print("=" * 50)

# For UCY, trajdata expects the original .vsp files, not converted ones
original_ucy_path = r"C:\Users\Jaima\Downloads\Socialzonenav\datasets\eth_ucy_peds\ucy"

# Copy original UCY .vsp files (not converted)
ucy_folders = ["zara01", "zara02", "zara03", "students01", "students03", "uni_examples", "arxiepiskopi"]

for folder in ucy_folders:
    src_folder = os.path.join(original_ucy_path, folder)
    dst_folder = os.path.join(native_path, folder)
    
    if os.path.exists(src_folder):
        # Copy the entire folder with original .vsp files
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
        print(f"✅ Copied {folder} with original .vsp files")

# For ETH, trajdata expects the original obsmat.txt files
original_eth_path = r"C:\Users\Jaima\Downloads\Socialzonenav\datasets\eth_ucy_peds\eth"

eth_folders = ["seq_eth", "seq_hotel"]
for folder in eth_folders:
    src_folder = os.path.join(original_eth_path, folder)
    dst_folder = os.path.join(native_path, folder)
    
    if os.path.exists(src_folder):
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
        print(f"✅ Copied {folder} with original obsmat.txt")

print(f"\n✅ Native files saved to: {native_path}")
print("\n" + "=" * 50)
print("TESTING WITH NATIVE FILES")
print("=" * 50)

# Now try loading with trajdata using the native format
try:
    dataset = UnifiedDataset(
        desired_data=[
            "eupeds_eth", "eupeds_hotel",
            "eupeds_zara1", "eupeds_zara2", "eupeds_zara3",
            "eupeds_univ", "eupeds_students"
        ],
        data_dirs={
            "eupeds_eth": native_path,
            "eupeds_hotel": native_path,
            "eupeds_zara1": native_path,
            "eupeds_zara2": native_path,
            "eupeds_zara3": native_path,
            "eupeds_univ": native_path,
            "eupeds_students": native_path,
        },
        desired_dt=0.4,
        standardize_data=True,
        verbose=True,
    )
    
    print(f"\n✅ SUCCESS! Loaded {len(dataset)} scenes")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    
    # If that fails, try with different dataset names
    print("\nTrying alternative dataset names...")
    
    alt_names = [
        "eth", "hotel", "zara1", "zara2", "zara3", "univ", "students"
    ]
    
    for name in alt_names:
        try:
            test = UnifiedDataset(
                desired_data=[name],
                data_dirs={name: native_path},
                desired_dt=0.4,
                verbose=False,
            )
            print(f"✅ {name} works!")
        except Exception as e:
            print(f"❌ {name}: {str(e)[:50]}")