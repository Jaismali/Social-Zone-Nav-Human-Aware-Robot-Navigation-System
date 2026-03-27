from trajdata import UnifiedDataset
import os

fixed_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_fixed"

print("=" * 50)
print("TESTING FIXED DATASET")
print("=" * 50)

# Try loading with correct names
try:
    dataset = UnifiedDataset(
        desired_data=[
            "eupeds_eth", "eupeds_hotel", 
            "eupeds_zara1", "eupeds_zara2", "eupeds_zara3",
            "eupeds_univ", "eupeds_students"
        ],
        data_dirs={
            "eupeds_eth": fixed_path,
            "eupeds_hotel": fixed_path,
            "eupeds_zara1": fixed_path,
            "eupeds_zara2": fixed_path,
            "eupeds_zara3": fixed_path,
            "eupeds_univ": fixed_path,
            "eupeds_students": fixed_path,
        },
        desired_dt=0.4,
        standardize_data=True,
        verbose=True,
    )
    
    print(f"\n✅ SUCCESS! Loaded {len(dataset)} scenes")
    print(f"Dataset contains {len(dataset)} trajectories")
    
    # Try to get first scene
    first_scene = dataset[0]
    print(f"\nFirst scene type: {type(first_scene)}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    
    # Try one by one to see which fails
    print("\nTrying one by one:")
    test_names = [
        "eupeds_eth", "eupeds_hotel", 
        "eupeds_zara1", "eupeds_zara2", "eupeds_zara3",
        "eupeds_univ", "eupeds_students"
    ]
    
    for name in test_names:
        try:
            test = UnifiedDataset(
                desired_data=[name],
                data_dirs={name: fixed_path},
                desired_dt=0.4,
                verbose=False,
            )
            print(f"  ✅ {name} works!")
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}")