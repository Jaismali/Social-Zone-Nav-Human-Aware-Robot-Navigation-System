from trajdata import UnifiedDataset
import trajdata

print("=" * 50)
print("TRAJDATA DATASET INFORMATION")
print("=" * 50)

# Try to get list of available datasets
try:
    from trajdata.dataset import DatasetType
    print("\nAvailable dataset types:")
    for dtype in DatasetType:
        print(f"  - {dtype.name}")
except:
    print("Could not import DatasetType")

print("\n" + "=" * 50)
print("TRYING DIFFERENT DATASET NAMES")
print("=" * 50)

dataset_path = r"C:\Users\Jaima\Downloads\eth_ucy_final_structure"

# Let's try ALL possible name variations
possible_names = [
    # Standard eupeds_* names
    "eupeds_eth", "eupeds_hotel", "eupeds_zara1", "eupeds_zara2", 
    "eupeds_zara3", "eupeds_univ", "eupeds_students",
    
    # Without eupeds_ prefix
    "eth", "hotel", "zara1", "zara2", "zara3", "univ", "students",
    
    # With ucy_ prefix
    "ucy_eth", "ucy_hotel", "ucy_zara1", "ucy_zara2", "ucy_zara3", 
    "ucy_univ", "ucy_students",
    
    # With biwi_ prefix for ETH
    "biwi_eth", "biwi_hotel",
    
    # Other variations from the error messages
    "zurich_eth", "zurich_hotel", "cyprus_zara1", "cyprus_zara2",
    "cyprus_zara3", "cyprus_univ", "cyprus_students"
]

for name in possible_names:
    try:
        print(f"\nTrying: '{name}'")
        test = UnifiedDataset(
            desired_data=[name],
            data_dirs={name: dataset_path},
            desired_dt=0.4,
            verbose=False,
        )
        print(f"  ✅ SUCCESS! {name} loaded {len(test)} scenes")
    except Exception as e:
        error_str = str(e)
        if "not supported" in error_str:
            print(f"  ❌ {name}: Not supported by trajdata")
        else:
            print(f"  ❌ {name}: {error_str[:100]}")