from trajdata import UnifiedDataset
import os

print("=" * 50)
print("CHECKING DATASET")
print("=" * 50)

dataset_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_final"
print(f"Looking for files in: {dataset_path}")
print("-" * 30)

# Check if path exists
if os.path.exists(dataset_path):
    print("✅ Folder found!")
    print("\nFiles in folder:")
    for f in os.listdir(dataset_path):
        print(f"  📄 {f}")
else:
    print("❌ Folder NOT found!")
    print("Please check if this path is correct:")
    print(dataset_path)
    exit()

print("\n" + "=" * 50)
print("TESTING DATASET LOADING")
print("=" * 50)

# Try loading with different names
test_names = [
    "eupeds_eth", 
    "eth", 
    "biwi_eth"
]

for name in test_names:
    print(f"\nTrying name: '{name}'")
    try:
        dataset = UnifiedDataset(
            desired_data=[name],
            data_dirs={name: dataset_path},
            desired_dt=0.4,
            verbose=True,
        )
        print(f"✅ SUCCESS with '{name}'!")
        break
    except Exception as e:
        print(f"❌ Failed with '{name}': {str(e)}")