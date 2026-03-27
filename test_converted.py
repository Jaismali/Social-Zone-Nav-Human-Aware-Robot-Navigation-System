from trajdata import UnifiedDataset
import os

converted_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_converted"

print("=" * 50)
print("TESTING CONVERTED DATASET")
print("=" * 50)

# Try loading with the correct names
try:
    dataset = UnifiedDataset(
        desired_data=[
            "eupeds_eth", "eupeds_hotel", 
            "eupeds_zara1", "eupeds_zara2", "eupeds_zara3",
            "eupeds_univ", "eupeds_students"
        ],
        data_dirs={
            "eupeds_eth": converted_path,
            "eupeds_hotel": converted_path,
            "eupeds_zara1": converted_path,
            "eupeds_zara2": converted_path,
            "eupeds_zara3": converted_path,
            "eupeds_univ": converted_path,
            "eupeds_students": converted_path,
        },
        desired_dt=0.4,
        standardize_data=True,
        verbose=True,
    )
    
    print(f"\n✅ SUCCESS! Loaded {len(dataset)} scenes")
    print(f"Dataset contains {len(dataset)} trajectories")
    
except Exception as e:
    print(f"❌ ERROR: {e}")