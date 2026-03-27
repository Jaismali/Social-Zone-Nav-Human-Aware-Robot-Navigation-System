import os

dataset_path = r"C:\Users\Jaima\Downloads\eth_ucy_dataset_final"

print("=" * 50)
print("CHECKING FILE FORMATS")
print("=" * 50)

# Check first few lines of each file type
files_to_check = [
    "biwi_eth.txt",
    "crowds_zara01.txt", 
    "students001.txt"
]

for filename in files_to_check:
    filepath = os.path.join(dataset_path, filename)
    print(f"\n📄 Checking: {filename}")
    print("-" * 30)
    
    if os.path.exists(filepath):
        # Show first 5 lines
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"Line {i+1}: {line.strip()}")
                else:
                    break
        print(f"\nFile size: {os.path.getsize(filepath)} bytes")
    else:
        print(f"❌ File not found: {filepath}")