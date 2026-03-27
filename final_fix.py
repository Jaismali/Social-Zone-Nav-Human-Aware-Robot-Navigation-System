import os
import shutil
import numpy as np
from trajdata import UnifiedDataset

# Create the exact folder structure trajdata expects
final_path = r"C:\Users\Jaima\Downloads\eth_ucy_final"
os.makedirs(final_path, exist_ok=True)

print("=" * 50)
print("CREATING FINAL DATASET STRUCTURE")
print("=" * 50)

# Source paths
converted_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_converted"
native_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_native"

# File mapping - trajdata expects these exact filenames in one folder
file_mapping = {
    # ETH files
    os.path.join(converted_path, "biwi_eth.txt"): os.path.join(final_path, "biwi_eth.txt"),
    os.path.join(converted_path, "biwi_hotel.txt"): os.path.join(final_path, "biwi_hotel.txt"),
    
    # UCY files - must be named exactly as trajdata expects
    os.path.join(converted_path, "crowds_zara01.txt"): os.path.join(final_path, "crowds_zara01.txt"),
    os.path.join(converted_path, "crowds_zara02.txt"): os.path.join(final_path, "crowds_zara02.txt"),
    os.path.join(converted_path, "crowds_zara03.txt"): os.path.join(final_path, "crowds_zara03.txt"),
    os.path.join(converted_path, "students001.txt"): os.path.join(final_path, "students001.txt"),
    os.path.join(converted_path, "students003.txt"): os.path.join(final_path, "students003.txt"),
    os.path.join(converted_path, "uni_examples.txt"): os.path.join(final_path, "uni_examples.txt"),
}

# Copy all files
for src, dst in file_mapping.items():
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✅ Copied {os.path.basename(src)}")
    else:
        print(f"❌ Missing: {os.path.basename(src)}")

print("\n" + "=" * 50)
print("TESTING WITH FINAL STRUCTURE")
print("=" * 50)

# Now test with the exact structure from documentation
try:
    dataset = UnifiedDataset(
        desired_data=[
            "eupeds_eth", "eupeds_hotel", 
            "eupeds_zara1", "eupeds_zara2", "eupeds_zara3",
            "eupeds_univ", "eupeds_students"
        ],
        data_dirs={
            "eupeds_eth": final_path,
            "eupeds_hotel": final_path,
            "eupeds_zara1": final_path,
            "eupeds_zara2": final_path,
            "eupeds_zara3": final_path,
            "eupeds_univ": final_path,
            "eupeds_students": final_path,
        },
        desired_dt=0.4,
        standardize_data=True,
        verbose=True,
    )
    
    print(f"\n✅ SUCCESS! Loaded {len(dataset)} scenes")
    print(f"First scene: {dataset[0]}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    
    # If that fails, let's try with relative paths
    print("\nTrying with relative paths...")
    try:
        # Change to the final directory
        os.chdir(final_path)
        
        dataset = UnifiedDataset(
            desired_data=[
                "eupeds_eth", "eupeds_hotel", 
                "eupeds_zara1", "eupeds_zara2", "eupeds_zara3",
                "eupeds_univ", "eupeds_students"
            ],
            data_dirs={
                "eupeds_eth": ".",
                "eupeds_hotel": ".",
                "eupeds_zara1": ".",
                "eupeds_zara2": ".",
                "eupeds_zara3": ".",
                "eupeds_univ": ".",
                "eupeds_students": ".",
            },
            desired_dt=0.4,
            standardize_data=True,
            verbose=True,
        )
        
        print(f"\n✅ SUCCESS with relative paths! Loaded {len(dataset)} scenes")
        
    except Exception as e2:
        print(f"\n❌ Still failing: {e2}")
        
        # List files in final folder
        print("\nFiles in final folder:")
        for f in os.listdir(final_path):
            size = os.path.getsize(os.path.join(final_path, f))
            print(f"  📄 {f} ({size} bytes)")