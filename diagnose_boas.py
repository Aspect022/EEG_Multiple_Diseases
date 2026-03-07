#!/usr/bin/env python3
"""
Deep diagnostic — traces EXACTLY why subjects fail.
Run: python diagnose_boas.py data/ds005555
"""
import sys, os, re
from pathlib import Path
import pandas as pd

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/ds005555")

# Discover subjects
subjects = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
print(f"Total subjects: {len(subjects)}")

has_psg_edf = 0
has_headband_edf = 0
has_psg_events = 0
has_headband_events = 0
has_stage_hum = 0
has_stage_ai_only = 0
no_edf = 0
no_events = 0
valid_epochs_count = 0
subjects_with_data = 0

for sub in subjects:
    sub_dir = data_dir / sub
    
    # Find EDF files
    psg_edfs = list(sub_dir.rglob('*psg*eeg*.edf'))
    hb_edfs = list(sub_dir.rglob('*headband*eeg*.edf'))
    all_edfs = list(sub_dir.rglob('*.edf'))
    
    if psg_edfs:
        has_psg_edf += 1
        edf_file = psg_edfs[0]
    elif all_edfs:
        # Pick non-headband, else headband
        non_hb = [f for f in all_edfs if 'headband' not in f.name.lower()]
        edf_file = non_hb[0] if non_hb else all_edfs[0]
        if hb_edfs:
            has_headband_edf += 1
    else:
        no_edf += 1
        print(f"  {sub}: NO EDF FILES AT ALL")
        continue
    
    # Find matching events.tsv
    events_name = re.sub(r'_eeg\.edf$', '_events.tsv', edf_file.name, flags=re.IGNORECASE)
    events_file = edf_file.parent / events_name
    
    if not events_file.exists():
        # Try broader search
        tsv_files = list(edf_file.parent.glob('*events*.tsv'))
        if tsv_files:
            events_file = tsv_files[0]
        else:
            no_events += 1
            print(f"  {sub}: EDF={edf_file.name} but NO EVENTS TSV (looked for {events_name})")
            # List what files DO exist
            all_files = [f.name for f in edf_file.parent.iterdir()]
            print(f"         Files: {all_files}")
            continue
    
    # Parse events
    try:
        df = pd.read_csv(events_file, sep='\t')
        cols = list(df.columns)
        
        # Check for stage columns
        if 'stage_hum' in cols:
            has_stage_hum += 1
            stage_col = 'stage_hum'
        elif 'stage_ai' in cols:
            has_stage_ai_only += 1
            stage_col = 'stage_ai'
        else:
            print(f"  {sub}: events has NO stage column! Columns: {cols}")
            continue
        
        # Count valid stages
        stages = df[stage_col].values
        valid = [(int(float(s))) for s in stages if 0 <= int(float(s)) <= 4]
        
        if valid:
            subjects_with_data += 1
            valid_epochs_count += len(valid)
        else:
            print(f"  {sub}: events found but ALL stages invalid. Unique values: {sorted(set(stages))}")
    
    except Exception as e:
        print(f"  {sub}: ERROR parsing {events_file.name}: {e}")

print(f"\n=== SUMMARY ===")
print(f"Subjects total:       {len(subjects)}")
print(f"With PSG EDF:         {has_psg_edf}")
print(f"With headband only:   {has_headband_edf}")
print(f"No EDF at all:        {no_edf}")
print(f"No events TSV:        {no_events}")
print(f"Has stage_hum col:    {has_stage_hum}")
print(f"Has stage_ai only:    {has_stage_ai_only}")
print(f"Subjects with data:   {subjects_with_data}")
print(f"Total valid epochs:   {valid_epochs_count}")
