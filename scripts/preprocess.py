#!/usr/bin/env python3
# scripts/preprocess.py

import os
import argparse
import numpy as np
import nibabel as nib
from nibabel.orientations import (
    io_orientation, axcodes2ornt, ornt_transform,
    apply_orientation, inv_ornt_aff
)
from scipy.ndimage import zoom
import SimpleITK as sitk


def unify_volume(vol: np.ndarray, target_shape, pad_value=None):
    """
    vol: (X, Y, Z) numpy array
    target_shape: tuple of desired (X, Y, Z)
    pad_value: intensity value to use for padding; if None, uses min(vol)
    Returns a volume cropped or padded to match target_shape.
    """
    if pad_value is None and vol.size:
        pad_value = float(np.min(vol))

    vol_out = vol
    for axis, tgt in enumerate(target_shape):
        curr = vol_out.shape[axis]
        diff = curr - tgt

        if diff > 0:
            # central crop
            start = diff // 2
            end = start + tgt
            sl = [slice(None)] * 3
            sl[axis] = slice(start, end)
            vol_out = vol_out[tuple(sl)]
        elif diff < 0:
            # pad with pad_value
            pad_before = (-diff) // 2
            pad_after = -diff - pad_before
            pad_width = [(0, 0)] * 3
            pad_width[axis] = (pad_before, pad_after)
            vol_out = np.pad(
                vol_out,
                pad_width,
                mode='constant',
                constant_values=pad_value
            )
    return vol_out


def reorient_to_RAS(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Reorient the given NIfTI image to RAS (Right-Anterior-Superior) orientation.
    """
    # current orientation
    ori = io_orientation(img.affine)
    # target orientation (RAS)
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))
    # compute transform
    transform = ornt_transform(ori, ras_ornt)
    # apply to data
    data2 = apply_orientation(img.get_fdata(), transform)
    # adjust affine
    affine2 = img.affine.dot(inv_ornt_aff(transform, img.shape))
    return nib.Nifti1Image(data2, affine2)


def resample_to_spacing(img, target_spacing=(1,1,1)):
    """Resample a NIfTI image to the given spacing (mm)"""
    data = img.get_fdata()
    orig_spacing = img.header.get_zooms()[:3]
    factors = [o/t for o, t in zip(orig_spacing, target_spacing)]
    resampled = zoom(data, factors, order=1)
    new_affine = img.affine.copy()
    new_affine[:3, :3] = np.diag(target_spacing)
    return nib.Nifti1Image(resampled, new_affine)


def zscore_normalize(x):
    """
    - If x is ndarray: return (x - mean) / std
    - If x is Nifti1Image: normalize its data and return new Nifti1Image
    """
    if isinstance(x, np.ndarray):
        arr = x
        m, s = arr.mean(), arr.std()
        return (arr - m) / (s + 1e-8)
    else:
        img = x
        data = img.get_fdata()
        m, s = data.mean(), data.std()
        norm = (data - m) / (s + 1e-8)
        return nib.Nifti1Image(norm, img.affine)


def find_series_dir(root, keyword):
    """Find first subdirectory under root containing keyword."""
    for r, dirs, _ in os.walk(root):
        for d in dirs:
            if keyword.lower() in d.lower():
                return os.path.join(r, d)
    raise FileNotFoundError(f"No series containing '{keyword}' under {root}")


def preprocess_btp(case_id, args):
    in_root = os.path.join(args.input, case_id)
    out_root = os.path.join(args.output, case_id)
    os.makedirs(out_root, exist_ok=True)

    seq_map = {
        "flair": "FLAIRreg",
        "t1":    "T1prereg",
        "t1ce":  "T1post",
        "t2":    "T2reg",
        "mask":  "MaskTumor",
    }

    for day in sorted(os.listdir(in_root)):
        in_day = os.path.join(in_root, day)
        if not os.path.isdir(in_day):
            continue

        out_day = os.path.join(out_root, day)
        os.makedirs(out_day, exist_ok=True)
        print(f"[{case_id}] 날짜: {day}")

        for name, key in seq_map.items():
            try:
                series_dir = find_series_dir(in_day, key)
            except FileNotFoundError:
                print(f"  - SKIP: '{key}' 없음")
                continue

            # 1) DICOM → (Z,Y,X)
            reader = sitk.ImageSeriesReader()
            files = reader.GetGDCMSeriesFileNames(series_dir)
            reader.SetFileNames(files)
            img_si = reader.Execute()
            arr_z_y_x = sitk.GetArrayFromImage(img_si)

            # 2) spacing correction
            sp_in = img_si.GetSpacing()[::-1]
            factors = [si/so for si, so in zip(sp_in, args.spacing)]
            arr_z_y_x = zoom(arr_z_y_x, factors, order=1)

            # 3) normalize (mask 제외)
            if name != "mask":
                arr_z_y_x = zscore_normalize(arr_z_y_x)

            # 4) axis transform → (X,Y,Z)
            arr_x_y_z = arr_z_y_x.transpose(2,1,0)

            # 5) unify shape
            arr_x_y_z = unify_volume(arr_x_y_z, tuple(args.target_shape))

            # 6) NIfTI create
            affine = np.diag(list(args.spacing) + [1])
            img_out = nib.Nifti1Image(arr_x_y_z, affine)

            # 7) reorient to RAS
            img_out = reorient_to_RAS(img_out)

            # 8) save
            out_path = os.path.join(out_day, f"{name}.nii.gz")
            nib.save(img_out, out_path)
            print(f"    ✔ {name} 저장, shape={arr_x_y_z.shape}")


def preprocess_brats(case_id, args):
    in_dir = os.path.join(args.input, case_id)
    out_dir = os.path.join(args.output, case_id)
    os.makedirs(out_dir, exist_ok=True)

    modalities = {
        "flair": f"{case_id}_flair.nii.gz",
        "t1":    f"{case_id}_t1.nii.gz",
        "t1ce":  f"{case_id}_t1ce.nii.gz",
        "t2":    f"{case_id}_t2.nii.gz",
        "mask":  f"{case_id}_seg.nii.gz",
    }

    for name, fname in modalities.items():
        img = nib.load(os.path.join(in_dir, fname))
        # 1) resample
        img = resample_to_spacing(img, args.spacing)
        # 2) normalize (mask 제외)
        if name != "mask":
            img = zscore_normalize(img)
        # 3) reorient to RAS
        img = reorient_to_RAS(img)
        # 4) save
        out_path = os.path.join(out_dir, f"{name}.nii.gz")
        nib.save(img, out_path)
        print(f"[BRATS] {case_id}/{name} → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["btp","brats21"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--spacing", nargs=3, type=float, default=[1,1,1],
                   help="Resample spacing as X Y Z (mm)")
    p.add_argument("--target_shape", nargs=3, type=int, default=[240,240,155],
                   help="Target shape (X Y Z) to unify BTP volumes")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for case in sorted(os.listdir(args.input)):
        if args.dataset == "btp" and case.startswith("PGBM-"):
            preprocess_btp(case, args)
        if args.dataset == "brats21" and case.startswith("BraTS2021"):
            preprocess_brats(case, args)


if __name__ == "__main__":
    main()