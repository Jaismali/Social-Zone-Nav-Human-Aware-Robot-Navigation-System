import os
import shutil

# Create the final folder
final_path = r"C:\Users\Jaima\Downloads\eth_ucy_final_structure"
os.makedirs(final_path, exist_ok=True)

print("=" * 50)
print("CREATING FINAL ETH/UCY STRUCTURE")
print("=" * 50)
print("Following trajdata documentation exactly [citation:3]")

# Source paths to your original raw files
eth_source = r"C:\Users\Jaima\Downloads\Socialzonenav\datasets\eth_ucy_peds\eth"
ucy_source = r"C:\Users\Jaima\Downloads\Socialzonenav\datasets\eth_ucy_peds\ucy"

# 1. Copy ETH files (obsmat.txt -> biwi_eth.txt and biwi_hotel.txt)
eth_seq_path = os.path.join(eth_source, "seq_eth", "obsmat.txt")
eth_hotel_path = os.path.join(eth_source, "seq_hotel", "obsmat.txt")

if os.path.exists(eth_seq_path):
    shutil.copy2(eth_seq_path, os.path.join(final_path, "biwi_eth.txt"))
    print("✅ Copied biwi_eth.txt (raw ETH sequence)")
else:
    print(f"❌ Missing: {eth_seq_path}")

if os.path.exists(eth_hotel_path):
    shutil.copy2(eth_hotel_path, os.path.join(final_path, "biwi_hotel.txt"))
    print("✅ Copied biwi_hotel.txt (raw Hotel sequence)")
else:
    print(f"❌ Missing: {eth_hotel_path}")

# 2. Copy UCY files (annotation.vsp -> crowds_*.txt and students*.txt)
ucy_mapping = {
    "zara01": "crowds_zara01.txt",
    "zara02": "crowds_zara02.txt",
    "zara03": "crowds_zara03.txt",
    "students01": "students001.txt",
    "students03": "students003.txt",
    "uni_examples": "uni_examples.txt",
}

for src_folder, dst_filename in ucy_mapping.items():
    src_file = os.path.join(ucy_source, src_folder, "annotation.vsp")
    dst_file = os.path.join(final_path, dst_filename)
    
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"✅ Copied {dst_filename} (raw .vsp from {src_folder})")
    else:
        print(f"❌ Missing: {src_file}")

print("\n" + "=" * 50)
print("FILES IN FINAL FOLDER:")
print("=" * 50)
for f in sorted(os.listdir(final_path)):
    size = os.path.getsize(os.path.join(final_path, f))
    print(f"  📄 {f} ({size} bytes)")

print("\n" + "=" * 50)
print("TESTING WITH TRAJDATA")
print("=" * 50)

# Now test loading with trajdata
from trajdata import UnifiedDataset

try:
    dataset = UnifiedDataset(
        desired_data=[
            "eupeds_eth",     # biwi_eth.txt
            "eupeds_hotel",   # biwi_hotel.txt
            "eupeds_zara1",   # crowds_zara01.txt
            "eupeds_zara2",   # crowds_zara02.txt
            "eupeds_zara3",   # crowds_zara03.txt
            "eupeds_univ",    # uni_examples.txt
            "eupeds_students" # students001.txt and students003.txt
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
    print(f"Dataset contains {len(dataset)} trajectories")
    
    # Show first scene info
    first_scene = dataset[0]
    print(f"\nFirst scene: {first_scene}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    
    # If it fails, let's see what trajdata is trying to load
    print("\nTrying one dataset at a time:")
    test_names = [
        ("eupeds_eth", "biwi_eth.txt"),
        ("eupeds_hotel", "biwi_hotel.txt"),
        ("eupeds_zara1", "crowds_zara01.txt"),
        ("eupeds_zara2", "crowds_zara02.txt"),
        ("eupeds_zara3", "crowds_zara03.txt"),
        ("eupeds_univ", "uni_examples.txt"),
    ]
    
    for ds_name, filename in test_names:
        try:
            print(f"\nTesting {ds_name} ({filename})...")
            test = UnifiedDataset(
                desired_data=[ds_name],
                data_dirs={ds_name: final_path},
                desired_dt=0.4,
                verbose=True,
            )
            print(f"  ✅ {ds_name} works!")
        except Exception as e2:
            print(f"  ❌ {ds_name}: {str(e2)}")