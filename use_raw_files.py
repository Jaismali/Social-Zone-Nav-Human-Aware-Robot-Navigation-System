from trajdata import UnifiedDataset
import os
import shutil

# Create a folder with the EXACT structure from the documentation
raw_path = r"C:\Users\Jaima\Downloads\eth_ucy_raw"
os.makedirs(raw_path, exist_ok=True)

print("=" * 50)
print("SETTING UP RAW DATASET FILES")
print("=" * 50)

# Source paths to your original files
eth_source = r"C:\Users\Jaima\Downloads\Socialzonenav\datasets\eth_ucy_peds\eth"
ucy_source = r"C:\Users\Jaima\Downloads\Socialzonenav\datasets\eth_ucy_peds\ucy"

# Copy ETH files (need to get the obsmat.txt files and rename them)
eth_seq_path = os.path.join(eth_source, "seq_eth", "obsmat.txt")
eth_hotel_path = os.path.join(eth_source, "seq_hotel", "obsmat.txt")

if os.path.exists(eth_seq_path):
    shutil.copy2(eth_seq_path, os.path.join(raw_path, "biwi_eth.txt"))
    print("✅ Copied biwi_eth.txt (raw ETH sequence)")

if os.path.exists(eth_hotel_path):
    shutil.copy2(eth_hotel_path, os.path.join(raw_path, "biwi_hotel.txt"))
    print("✅ Copied biwi_hotel.txt (raw Hotel sequence)")

# Copy UCY raw .vsp files (NOT converted ones)
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
    dst_file = os.path.join(raw_path, dst_filename)
    
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"✅ Copied {dst_filename} (raw .vsp from {src_folder})")
    else:
        print(f"❌ Missing: {src_file}")

print("\n" + "=" * 50)
print("FILES IN RAW FOLDER:")
print("=" * 50)
for f in os.listdir(raw_path):
    size = os.path.getsize(os.path.join(raw_path, f))
    print(f"  📄 {f} ({size} bytes)")

print("\n" + "=" * 50)
print("TESTING WITH RAW FILES")
print("=" * 50)

# Test loading with raw files
try:
    dataset = UnifiedDataset(
        desired_data=[
            "eupeds_eth", "eupeds_hotel", 
            "eupeds_zara1", "eupeds_zara2", "eupeds_zara3",
            "eupeds_univ", "eupeds_students"
        ],
        data_dirs={
            "eupeds_eth": raw_path,
            "eupeds_hotel": raw_path,
            "eupeds_zara1": raw_path,
            "eupeds_zara2": raw_path,
            "eupeds_zara3": raw_path,
            "eupeds_univ": raw_path,
            "eupeds_students": raw_path,
        },
        desired_dt=0.4,
        standardize_data=True,
        verbose=True,
    )
    
    print(f"\n✅ SUCCESS! Loaded {len(dataset)} scenes")
    print(f"First scene: {dataset[0]}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    
    # Try one by one to identify which works
    print("\nTesting each dataset individually:")
    test_names = [
        "eupeds_eth", "eupeds_hotel", 
        "eupeds_zara1", "eupeds_zara2", "eupeds_zara3",
        "eupeds_univ", "eupeds_students"
    ]
    
    for name in test_names:
        try:
            test = UnifiedDataset(
                desired_data=[name],
                data_dirs={name: raw_path},
                desired_dt=0.4,
                verbose=False,
            )
            print(f"  ✅ {name} works!")
        except Exception as e2:
            print(f"  ❌ {name}: {str(e2)[:80]}")