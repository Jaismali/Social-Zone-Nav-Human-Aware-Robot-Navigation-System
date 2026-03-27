import trajdata
from trajdata import UnifiedDataset
import os

print(f"trajdata version: {trajdata.__version__}")
print("=" * 50)

dataset_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_final"
print(f"Checking path: {dataset_path}")
if os.path.exists(dataset_path):
    print("✅ Path exists!")
    print("\nFiles in directory:")
    for f in os.listdir(dataset_path):
        print(f"  - {f}")
else:
    print("❌ Path does not exist!")

print("\n" + "=" * 50)
print("Trying to load datasets...")

# Try each dataset name
names_to_try = [
    "eupeds_eth", "eupeds_hotel", "eupeds_zara1", 
    "eupeds_zara2", "eupeds_zara3", "eupeds_univ", 
    "eupeds_students"
]

for name in names_to_try:
    try:
        test = UnifiedDataset(
            desired_data=[name],
            data_dirs={name: dataset_path},
            desired_dt=0.4,
            verbose=False,
        )
        print(f"✅ {name} - WORKS!")
    except Exception as e:
        print(f"❌ {name} - FAILED: {str(e)}")