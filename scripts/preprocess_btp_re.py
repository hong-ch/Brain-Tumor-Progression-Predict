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

def unify_inplane(vol: np.ndarray, target_shape, pad_value=None):
    """
    Crop or pad each axial slice of `vol` (shape Z, Y, X) to exactly (Y_tgt, X_tgt).
    Z dimension (axis=0) is left unchanged.
    """
    if pad_value is None and vol.size:
        pad_value = float(vol.min())
    out = vol
    # adjust axis 1 (Y) and 2 (X) only
    for axis, tgt in zip((1, 2), target_shape):
        curr = out.shape[axis]
        diff = curr - tgt
        if diff > 0:
            start = diff // 2
            sl = [slice(None)] * 3
            sl[axis] = slice(start, start + tgt)
            out = out[tuple(sl)]
        elif diff < 0:
            pad_before = (-diff) // 2
            pad_after = -diff - pad_before
            pad_width = [(0,0)] * 3
            pad_width[axis] = (pad_before, pad_after)
            out = np.pad(out, pad_width, mode='constant', constant_values=pad_value)
    return out

def preprocess_btp(input_root, output_root, spacing, target_shape, target_z):
    """
    Process all BTP cases under input_root, writing outputs under output_root.
    - DICOM → NumPy (Z,Y,X)
    - Resample in-plane to spacing X,Y (mm), keep original Z count
    - Z‐score normalize (skip mask)
    - Crop/pad in-plane to (Y,X)
    - Crop/pad Z to target_z slices (central)
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

                # 2) Resample in-plane only (keep Z)
                orig_sp = img_si.GetSpacing()  # (X, Y, Z)
                fx = orig_sp[0] / spacing[0]
                fy = orig_sp[1] / spacing[1]
                arr = zoom(arr, (1.0, fy, fx), order=1)

                # 3) Z‐score normalize (skip mask)
                if name != "mask":
                    m, s = arr.mean(), arr.std()
                    arr = (arr - m) / (s + 1e-8)

                # 4) Crop/pad in-plane only (Y, X), keep Z unchanged
                arr = unify_inplane(arr, target_shape)

                # 5) Crop/pad Z to target_z (central)
                z_curr = arr.shape[0]
                pad_val = float(arr.min()) if arr.size else 0
                if z_curr > target_z:
                    start = (z_curr - target_z) // 2
                    arr = arr[start:start+target_z, :, :]
                elif z_curr < target_z:
                    pad_before = (target_z - z_curr) // 2
                    pad_after  = target_z - z_curr - pad_before
                    arr = np.pad(
                        arr,
                        ((pad_before, pad_after), (0,0), (0,0)),
                        mode='constant',
                        constant_values=pad_val
                    )

                # 6) Save as NIfTI
                affine = np.diag([spacing[0], spacing[1], orig_sp[2], 1])
                img_out = nib.Nifti1Image(arr.astype(np.float32), affine)
                out_path = os.path.join(date_out, f"{name}.nii.gz")
                nib.save(img_out, out_path)

                print(f"    ✔ {name} saved, shape={arr.shape}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocess BTP-only: 2D in-plane resample + Z-slices fixed"
    )
    p.add_argument("--input",       required=True,
                   help="Root folder of raw BTP DICOM cases")
    p.add_argument("--output",      required=True,
                   help="Output folder for preprocessed NIfTI files")
    p.add_argument("--spacing",     nargs=2, type=float, default=[1,1],
                   help="In-plane spacing (X Y) to resample to, mm")
    p.add_argument("--target_shape",nargs=2, type=int, default=[240,240],
                   help="Desired in-plane shape (Y X)")
    p.add_argument("--target_z",    type=int, default=22,
                   help="Desired number of Z slices after crop/pad")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    preprocess_btp(
        input_root   = args.input,
        output_root  = args.output,
        spacing      = tuple(args.spacing),
        target_shape = tuple(args.target_shape),
        target_z     = args.target_z
    )
