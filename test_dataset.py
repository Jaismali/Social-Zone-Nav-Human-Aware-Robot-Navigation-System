import os
from trajdata import UnifiedDataset

dataset_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_final"

print("=" * 50)
print("TESTING ETH/UCY DATASET")
print("=" * 50)

# Try with corrected dataset names
try:
    dataset = UnifiedDataset(
        desired_data=[
            "eth",  # Instead of eupeds_eth
            "hotel",  # Instead of eupeds_hotel
            "zara1",  # Instead of eupeds_zara1
            "zara2",  # Instead of eupeds_zara2
            "zara3",  # Instead of eupeds_zara3
            "univ",  # Instead of eupeds_univ
            "students"  # Instead of eupeds_students
        ],
        data_dirs={
            "eth": dataset_path,
            "hotel": dataset_path,
            "zara1": dataset_path,
            "zara2": dataset_path,
            "zara3": dataset_path,
            "univ": dataset_path,
            "students": dataset_path,
        },
        desired_dt=0.4,
        standardize_data=True,
        verbose=True,
    )

    print(f"\n✅ SUCCESS! Loaded {len(dataset)} scenes")

except Exception as e:
    print(f"❌ ERROR: {e}")

    # If that fails, let's try one by one to find which works
    print("\nTrying datasets one by one:")
    test_names = ["eth", "hotel", "zara1", "zara2", "zara3", "univ", "students"]
    for name in test_names:
        try:
            test = UnifiedDataset(
                desired_data=[name],
                data_dirs={name: dataset_path},
                verbose=False,
            )
            print(f"  ✅ {name} works!")
        except Exception as e:
            print(f"  ❌ {name}: {e}")