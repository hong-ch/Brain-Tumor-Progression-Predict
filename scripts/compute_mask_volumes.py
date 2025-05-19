#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import nibabel as nib

def compute_mask_volumes(preproc_root, output_csv):
    records = []
    for case in sorted(os.listdir(preproc_root)):
        case_dir = os.path.join(preproc_root, case)
        if not os.path.isdir(case_dir):
            continue

        for date in sorted(os.listdir(case_dir)):
            date_dir = os.path.join(case_dir, date)
            if not os.path.isdir(date_dir):
                continue

            mask_path = os.path.join(date_dir, 'mask.nii.gz')
            if not os.path.exists(mask_path):
                print(f"[WARNING] mask not found: {mask_path}")
                continue

            # 1) load mask and count voxels
            img = nib.load(mask_path)
            data = img.get_fdata()
            voxel_count = (data > 0).sum()

            # 2) get voxel size in mm (spacing)
            zooms = img.header.get_zooms()[:3]  # (dx, dy, dz) in mm
            voxel_volume = zooms[0] * zooms[1] * zooms[2]

            # 3) compute volumes
            volume_mm3 = voxel_count * voxel_volume
            volume_ml  = volume_mm3 / 1000.0

            records.append({
                'PatientID': case,
                'Date': date,
                'MaskVoxelCount': voxel_count,
                'MaskVolume_mm3': volume_mm3,
                'MaskVolume_mL': volume_ml
            })

    # save to CSV
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"→ Saved mask volumes for {len(df)} entries to {output_csv}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Compute mask volumes from preprocessed BTP data"
    )
    p.add_argument(
        "--preproc_root",
        default="../data/btp_preproc_new",
        help="Path to preprocessed BTP root"
    )
    p.add_argument(
        "--output_csv",
        default="../csv/btp_mask_volumes.csv",
        help="Where to write the CSV of mask volumes"
    )
    args = p.parse_args()

    compute_mask_volumes(args.preproc_root, args.output_csv)
