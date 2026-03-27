from trajdata import UnifiedDataset
import os

converted_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_converted"

print("=" * 50)
print("FINDING CORRECT DATASET NAMES")
print("=" * 50)

# List all possible name combinations to try
possible_names = [
    # With eupeds_ prefix
    ["eupeds_eth", "eupeds_hotel", "eupeds_zara1", "eupeds_zara2", "eupeds_zara3", "eupeds_univ", "eupeds_students"],
    # Without prefix
    ["eth", "hotel", "zara1", "zara2", "zara3", "univ", "students"],
    # With ucy_ prefix
    ["ucy_eth", "ucy_hotel", "ucy_zara1", "ucy_zara2", "ucy_zara3", "ucy_univ", "ucy_students"],
    # With biwi_ prefix for ETH
    ["biwi_eth", "biwi_hotel", "zara1", "zara2", "zara3", "univ", "students"],
]

for name_set in possible_names:
    print(f"\nTrying name set: {name_set}")
    print("-" * 30)
    
    try:
        # Create data_dirs dictionary
        data_dirs = {}
        for name in name_set:
            data_dirs[name] = converted_path
        
        dataset = UnifiedDataset(
            desired_data=name_set,
            data_dirs=data_dirs,
            desired_dt=0.4,
            standardize_data=True,
            verbose=False,  # Set to False to reduce output
        )
        print(f"✅ SUCCESS! Loaded {len(dataset)} scenes")
        print(f"Working names: {name_set}")
        break
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}...")  # Show first 100 chars of error

print("\n" + "=" * 50)
print("Testing individual datasets:")
print("=" * 50)

# Test each possible name individually
individual_names = [
    "eupeds_eth", "eth", "biwi_eth",
    "eupeds_hotel", "hotel", "biwi_hotel",
    "eupeds_zara1", "zara1",
    "eupeds_zara2", "zara2",
    "eupeds_zara3", "zara3",
    "eupeds_univ", "univ",
    "eupeds_students", "students"
]

for name in individual_names:
    try:
        test = UnifiedDataset(
            desired_data=[name],
            data_dirs={name: converted_path},
            desired_dt=0.4,
            verbose=False,
        )
        print(f"✅ {name} works!")
    except Exception as e:
        print(f"❌ {name}: {str(e)[:50]}")