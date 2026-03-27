"""
download_datasets.py
Downloads ETH and UCY pedestrian trajectory datasets.
Data is stored in ./data/ directory.
"""

import os
import requests
import zipfile
import io

# ETH/UCY datasets are available via these public repositories
# Using the preprocessed versions from Social Force / LSTM trajectory prediction papers
DATASET_URLS = {
    "eth": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/eth/test/biwi_eth.txt",
    "hotel": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/hotel/test/biwi_hotel.txt",
    "univ": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/univ/test/students003.txt",
    "zara1": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/zara1/test/crowds_zara01.txt",
    "zara2": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/zara2/test/crowds_zara02.txt",
}

# Also include train splits
DATASET_URLS_TRAIN = {
    "eth_train": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/eth/train/biwi_eth.txt",
    "hotel_train": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/hotel/train/biwi_hotel.txt",
    "univ_train": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/univ/train/students003.txt",
    "zara1_train": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/zara1/train/crowds_zara01.txt",
    "zara2_train": "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/zara2/train/crowds_zara02.txt",
}


def download_file(url, dest_path):
    """Download a single file from URL to dest_path."""
    print(f"  Downloading {os.path.basename(dest_path)} from {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(resp.content)
        print(f"  -> Saved {os.path.basename(dest_path)} ({len(resp.content)} bytes)")
        return True
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return False


def generate_synthetic_data(dest_path, n_agents=20, n_frames=200, seed=42):
    """
    Generate synthetic trajectory data mimicking ETH/UCY format when download fails.
    Format: frame_id  agent_id  x  y
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    lines = []
    # Agents walk in roughly straight lines with some noise
    starts = rng.uniform(-5, 5, (n_agents, 2))
    velocities = rng.uniform(-0.05, 0.05, (n_agents, 2))
    velocities += rng.choice([-1, 1], size=(n_agents, 2)) * rng.uniform(0.02, 0.08, (n_agents, 2))

    for frame in range(n_frames):
        for agent_id in range(n_agents):
            pos = starts[agent_id] + velocities[agent_id] * frame
            pos += rng.normal(0, 0.005, 2)  # small noise
            lines.append(f"{frame}\t{agent_id}\t{pos[0]:.4f}\t{pos[1]:.4f}")

    with open(dest_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  -> Generated synthetic data: {dest_path}")


def download_datasets(data_dir="data"):
    """Download all ETH/UCY datasets."""
    os.makedirs(data_dir, exist_ok=True)

    all_urls = {**DATASET_URLS, **DATASET_URLS_TRAIN}
    success_count = 0

    print("Downloading ETH/UCY pedestrian datasets...")
    print("=" * 60)

    for name, url in all_urls.items():
        dest = os.path.join(data_dir, f"{name}.txt")
        if os.path.exists(dest):
            print(f"  {name}.txt already exists, skipping.")
            success_count += 1
            continue
        ok = download_file(url, dest)
        if ok:
            success_count += 1

    # If downloads failed, generate synthetic data
    synth_files = [
        ("eth_synth", 42),
        ("hotel_synth", 43),
        ("univ_synth", 44),
    ]
    for name, seed in synth_files:
        dest = os.path.join(data_dir, f"{name}.txt")
        if not os.path.exists(dest):
            print(f"  Generating synthetic dataset: {name}")
            generate_synthetic_data(dest, n_agents=15, n_frames=300, seed=seed)

    print(f"\nDownload complete. {success_count}/{len(all_urls)} files downloaded.")
    print(f"Data directory: {os.path.abspath(data_dir)}")
    return data_dir


if __name__ == "__main__":
    download_datasets()
