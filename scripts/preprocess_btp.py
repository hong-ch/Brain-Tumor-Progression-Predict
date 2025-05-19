#!/usr/bin/env python3
# scripts/preprocess_btp_only.py

import os
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom

def find_series_dir(root, keyword):
    """Find first subdirectory under root whose name contains keyword (case‐insensitive)."""
    for dirpath, dirs, _ in os.walk(root):
        for d in dirs:
            if keyword.lower() in d.lower():
                return os.path.join(dirpath, d)
    raise FileNotFoundError(f"No series containing '{keyword}' under {root}")

def unify_volume(vol: np.ndarray, target_shape, pad_value=None):
    """
    Crop or pad `vol` to exactly `target_shape`.
    If pad_value is None, use the volume's min as padding (preserves background).
    """
    if pad_value is None and vol.size:
        pad_value = float(vol.min())

    out = vol
    for axis, tgt in enumerate(target_shape):
        curr = out.shape[axis]
        diff = curr - tgt

        if diff > 0:
            # central crop
            start = diff // 2
            sl = [slice(None)] * 3
            sl[axis] = slice(start, start + tgt)
            out = out[tuple(sl)]
        elif diff < 0:
            # pad with pad_value
            pad_before = (-diff) // 2
            pad_after = -diff - pad_before
            pad_width = [(0, 0)] * 3
            pad_width[axis] = (pad_before, pad_after)
            out = np.pad(
                out,
                pad_width,
                mode='constant',
                constant_values=pad_value
            )
    return out

def preprocess_btp(input_root, output_root, spacing, target_shape):
    """
    Process all BTP cases under input_root, writing outputs under output_root.
    - Converts DICOM → NumPy (Z,Y,X)
    - Resamples to `spacing` (X,Y,Z) in mm
    - Z‐score normalizes (skip mask)
    - Crops/pads to `target_shape` (Z,Y,X)
    """
    seq_map = {
        "flair": "FLAIRreg",
        "t1":    "T1prereg",
        "t1ce":  "T1post",
        "t2":    "T2reg",
        "mask":  "MaskTumor",
    }

    for case in sorted(os.listdir(input_root)):
        case_in = os.path.join(input_root, case)
        if not os.path.isdir(case_in):
            continue
        case_out = os.path.join(output_root, case)
        os.makedirs(case_out, exist_ok=True)

        for date in sorted(os.listdir(case_in)):
            date_in = os.path.join(case_in, date)
            if not os.path.isdir(date_in):
                continue
            date_out = os.path.join(case_out, date)
            os.makedirs(date_out, exist_ok=True)

            print(f"[{case}] Processing session {date}")
            for name, key in seq_map.items():
                try:
                    series_dir = find_series_dir(date_in, key)
                except FileNotFoundError:
                    print(f"  - SKIP {name}: no folder matching '{key}'")
                    continue

                # 1) Read DICOM series → NumPy (Z, Y, X)
                reader = sitk.ImageSeriesReader()
                files  = reader.GetGDCMSeriesFileNames(series_dir)
                reader.SetFileNames(files)
                img_si = reader.Execute()
                arr = sitk.GetArrayFromImage(img_si)  # shape: (Z, Y, X)

                # 2) Resample to target spacing
                #    img_si.GetSpacing() is (x, y, z)
                orig_sp = img_si.GetSpacing()[::-1]   # now (z, y, x)
                tgt_sp  = spacing[::-1]               # user passes (x, y, z)
                factors = [o/t for o, t in zip(orig_sp, tgt_sp)]
                arr = zoom(arr, factors, order=1)

                # 3) Z‐score normalize (skip mask)
                if name != "mask":
                    m, s = arr.mean(), arr.std()
                    arr = (arr - m) / (s + 1e-8)

                # 4) Crop/pad to target_shape (Z, Y, X)
                arr = unify_volume(arr, target_shape)

                # 5) Save as NIfTI
                affine = np.diag(list(spacing) + [1])  # identity-affine with correct spacing
                img_out = nib.Nifti1Image(arr.astype(np.float32), affine)
                out_path = os.path.join(date_out, f"{name}.nii.gz")
                nib.save(img_out, out_path)

                print(f"    ✔ {name} saved, shape={arr.shape}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocess BTP-only: DICOM→NIfTI, resample, normalize, unify shape"
    )
    p.add_argument("--input",        required=True,
                   help="Root folder of raw BTP cases (DICOM input)")
    p.add_argument("--output",       required=True,
                   help="Where to write preprocessed NIfTI files")
    p.add_argument("--spacing",      nargs=3, type=float, default=[1,1,1],
                   help="Voxel spacing to resample to (Z Y X) in mm")
    p.add_argument("--target_shape", nargs=3, type=int, default=[155,240,240],
                   help="Desired volume shape (Z Y X) after crop/pad")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    preprocess_btp(
        input_root   = args.input,
        output_root  = args.output,
        spacing      = tuple(args.spacing),
        target_shape = tuple(args.target_shape)
    )
