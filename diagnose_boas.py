#!/usr/bin/env python3
"""
Diagnostic script to inspect BOAS dataset structure.
Run on server: python diagnose_boas.py data/ds005555
"""
import sys
import os
from pathlib import Path

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ds005555"
data_path = Path(data_dir)

print(f"=== BOAS Dataset Diagnostic ===")
print(f"Path: {data_path}")
print(f"Exists: {data_path.exists()}")

# 1. Show top-level structure
print(f"\n--- Top-level dirs (first 5) ---")
subjects = sorted([d.name for d in data_path.iterdir() if d.is_dir() and d.name.startswith('sub-')])
print(f"Total subjects: {len(subjects)}")
for s in subjects[:5]:
    print(f"  {s}/")

# 2. Deep-dive into first subject
if subjects:
    sub = subjects[0]
    sub_dir = data_path / sub
    print(f"\n--- Full tree of {sub}/ ---")
    for root, dirs, files in os.walk(sub_dir):
        level = root.replace(str(sub_dir), '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            print(f"{indent}  {f}")

    # 3. Find and inspect EDF files
    edf_files = list(sub_dir.rglob("*.edf"))
    print(f"\n--- EDF files in {sub} ---")
    for ef in edf_files:
        print(f"  {ef.relative_to(data_path)}")

    # 4. Find and inspect events TSV files
    tsv_files = list(sub_dir.rglob("*events*"))
    print(f"\n--- Events files in {sub} ---")
    for tf in tsv_files:
        print(f"  {tf.relative_to(data_path)}")
        if tf.suffix == '.tsv':
            print(f"  Contents (first 10 rows):")
            with open(tf) as f:
                for i, line in enumerate(f):
                    if i < 10:
                        print(f"    {line.rstrip()}")
                    else:
                        break
            # Count total rows
            with open(tf) as f:
                total = sum(1 for _ in f) - 1  # minus header
            print(f"  Total rows: {total}")

    # 5. Load first EDF with MNE and check channels + annotations
    if edf_files:
        # Pick the PSG file if available
        psg_files = [f for f in edf_files if 'headband' not in f.name.lower()]
        target = psg_files[0] if psg_files else edf_files[0]

        print(f"\n--- MNE inspection of {target.name} ---")
        try:
            import mne
            mne.set_log_level('ERROR')
            raw = mne.io.read_raw_edf(str(target), preload=False, verbose=False)
            print(f"  Channels ({len(raw.ch_names)}): {raw.ch_names}")
            print(f"  Sampling rate: {raw.info['sfreq']} Hz")
            print(f"  Duration: {raw.times[-1]:.1f} seconds")

            # Check annotations
            annots = raw.annotations
            print(f"\n  Annotations count: {len(annots)}")
            if len(annots) > 0:
                # Show unique descriptions
                descs = [a['description'] for a in annots]
                unique_descs = sorted(set(descs))
                print(f"  Unique annotation labels ({len(unique_descs)}):")
                for d in unique_descs:
                    count = descs.count(d)
                    print(f"    '{d}' -> {count} occurrences")

                # Show first 5 annotations
                print(f"\n  First 5 annotations:")
                for i, a in enumerate(annots):
                    if i >= 5:
                        break
                    print(f"    onset={a['onset']:.1f}s, dur={a['duration']:.1f}s, desc='{a['description']}'")
        except Exception as e:
            print(f"  MNE Error: {e}")

print("\n=== Done ===")
