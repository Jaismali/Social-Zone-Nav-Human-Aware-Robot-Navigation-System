"""
prepare_real_data.py
====================
Copies raw ETH/UCY trajectory files from their original download locations
into the data/ directory used by extract_zones.py and train_zone_learner.py.

Run this once after downloading the raw datasets before running the main
pipeline.  Subsequent runs are idempotent (existing files are not overwritten
unless --force is passed).

Expected source layout
~~~~~~~~~~~~~~~~~~~~~~
::

    datasets/eth_ucy_peds/
    ├── eth/
    │   ├── seq_eth/obsmat.txt          ETH pedestrian sequence
    │   └── seq_hotel/obsmat.txt        Hotel sequence
    └── ucy/
        ├── students03/students003.txt  UCY University sequence
        ├── zara01/obsmat.txt           Zara 01 sequence
        └── zara02/obsmat.txt           Zara 02 sequence

File formats
~~~~~~~~~~~~
ETH sequences (obsmat.txt): 8 columns, space/tab delimited::

    frame_id  agent_id  pos_x  pos_z  pos_y  vel_x  vel_z  vel_y

The coordinate system is world-frame with y as the vertical axis; the
horizontal plane is (pos_x, pos_y) — note pos_z is the vertical component
and is discarded.  Positions are in metres.

UCY students003.txt: 4 columns (frame_id, agent_id, x, y) — no velocity,
no vertical component.  Already in the format expected by extract_zones.py.

UCY zara01/zara02 obsmat.txt: same 8-column format as ETH.

All five files are renamed on copy to avoid ambiguity (two different
sequences share the filename "obsmat.txt").

Real data extraction results (443,074 agent pairs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Running extract_zones.py on these five files produced:

    Zone    Raw 10th-pct    Interpretation
    front   0.436 m         very tight — human-human crowding
    back    0.412 m         similarly tight
    left    0.967 m
    right   0.929 m

These raw values reflect *human-to-human* comfortable distances, which are
substantially smaller than *human-to-robot* distances because people naturally
give robots more clearance and react less predictably to them (Syrdal et al.,
2007; Mead & Mataric, 2017).  scale_zones.py applies empirically-motivated
scaling factors to convert to robot-appropriate thresholds.

References
----------
Pellegrini, S., Ess, A., Schindler, K., & Van Gool, L. (2009). You'll never
    walk alone. In Proc. ICCV, pp. 261-268.  ETH dataset.

Lerner, A., Chrysanthou, Y., & Lischinski, D. (2007). Crowds by example.
    Computer Graphics Forum, 26(3), 655-664.  UCY dataset.

Syrdal, D. S., Dautenhahn, K., Walters, M. L., & Koay, K. L. (2007).
    Robot personal space — proxemics for human-robot interaction.
    In AISB'07 Workshop on Perspectives on How Humans and Robots Interact.
    Empirical evidence that robots require ~2× larger personal space than
    humans expect from other humans.

Mead, R., & Mataric, M. J. (2017). Autonomous human-robot proxemics: social
    compliance and complaisance. In Int. J. Social Robotics, 9(2), 159-170.
"""

import argparse
import os
import shutil


# ---------------------------------------------------------------------------
# Source → destination file mapping
# ---------------------------------------------------------------------------
# Paths use os.path.join at runtime to handle Windows/Linux separators.
# Source paths are relative to the directory where this script is run.

FILE_MAP = [
    # (source_relative_path,                                dest_filename)
    (os.path.join('datasets', 'eth_ucy_peds', 'eth', 'seq_eth',   'obsmat.txt'),   'eth_obsmat.txt'),
    (os.path.join('datasets', 'eth_ucy_peds', 'eth', 'seq_hotel', 'obsmat.txt'),   'hotel_obsmat.txt'),
    (os.path.join('datasets', 'eth_ucy_peds', 'ucy', 'students03','students003.txt'), 'students003.txt'),
    (os.path.join('datasets', 'eth_ucy_peds', 'ucy', 'zara01',    'obsmat.txt'),   'zara01_obsmat.txt'),
    (os.path.join('datasets', 'eth_ucy_peds', 'ucy', 'zara02',    'obsmat.txt'),   'zara02_obsmat.txt'),
]


def prepare_real_data(data_dir='data', force=False):
    """
    Copy raw ETH/UCY files into data_dir with unambiguous names.

    Parameters
    ----------
    data_dir : str
        Destination directory (created if absent).
    force : bool
        If True, overwrite existing destination files.

    Returns
    -------
    list of str
        Destination paths of successfully copied files.
    """
    os.makedirs(data_dir, exist_ok=True)
    copied = []
    missing_src = []

    print(f"Preparing ETH/UCY data files into '{data_dir}/'")
    print("=" * 60)

    for src, dst_name in FILE_MAP:
        dst = os.path.join(data_dir, dst_name)

        if not os.path.exists(src):
            print(f"  MISSING  {src}")
            missing_src.append(src)
            continue

        if os.path.exists(dst) and not force:
            size = os.path.getsize(dst)
            print(f"  EXISTS   {dst_name}  ({size:,} bytes) — skipping")
            copied.append(dst)
            continue

        shutil.copy2(src, dst)
        size = os.path.getsize(dst)
        print(f"  Copied   {src}")
        print(f"        -> {dst}  ({size:,} bytes)")
        copied.append(dst)

    print()
    print(f"Files ready: {len(copied)}/{len(FILE_MAP)}")

    if missing_src:
        print(f"\nWARNING: {len(missing_src)} source file(s) not found:")
        for p in missing_src:
            print(f"  {p}")
        print("\nCheck that the datasets/ directory exists and matches the")
        print("expected layout shown in this file's module docstring.")

    if len(copied) == len(FILE_MAP):
        print("\nAll files present. Run next:")
        print("  python extract_zones.py")

    return copied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy ETH/UCY trajectory files into data/ for the pipeline."
    )
    parser.add_argument('--data-dir', default='data',
                        help='Destination directory (default: data/)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing destination files')
    args = parser.parse_args()
    prepare_real_data(data_dir=args.data_dir, force=args.force)
