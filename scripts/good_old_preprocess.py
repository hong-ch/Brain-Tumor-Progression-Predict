#!/usr/bin/env python3
# scripts/preprocess.py

import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk

def resample_to_spacing(img, target_spacing=(1,1,1)):
    data = img.get_fdata()
    orig_spacing = img.header.get_zooms()[:3]
    factors = [o/t for o,t in zip(orig_spacing, target_spacing)]
    resampled = zoom(data, factors, order=1)  # linear interp
    new_affine = np.copy(img.affine)
    new_affine[:3, :3] = np.diag(target_spacing)
    return nib.Nifti1Image(resampled, new_affine)

def zscore_normalize(img):
    data = img.get_fdata()
    m, s = data.mean(), data.std()
    norm = (data - m) / (s + 1e-8)
    return nib.Nifti1Image(norm, img.affine)

def rotate_left(img):
    """in-plane 90° left rotation (axes 0↔1)"""
    data = img.get_fdata()
    rot = np.rot90(data, k=1, axes=(0,1))
    # adjust affine: swap and negate axes in affine matrix
    aff = img.affine.copy()
    # build a rotation matrix for 90° left
    R = np.array([[0,1,0,0],
                  [-1,0,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
    new_affine = aff @ R
    return nib.Nifti1Image(rot, new_affine)

def preprocess_brats(case_id, args):
    in_dir  = os.path.join(args.input, case_id)
    out_dir = os.path.join(args.output, case_id)
    os.makedirs(out_dir, exist_ok=True)

    # Brats 파일명 패턴
    modalities = {
        "flair":  f"{case_id}_flair.nii.gz",
        "t1":     f"{case_id}_t1.nii.gz",
        "t1ce":   f"{case_id}_t1ce.nii.gz",
        "t2":     f"{case_id}_t2.nii.gz",
        "mask":   f"{case_id}_seg.nii.gz",
    }

    for name, fname in modalities.items():
        in_path = os.path.join(in_dir, fname)
        img = nib.load(in_path)
        # 1) Resample 모두
        img = resample_to_spacing(img, args.spacing)
        # 2) Normalize (mask 제외)
        if name != "mask":
            img = zscore_normalize(img)
        # 3) Orientation 보정: Brats 볼륨은 왼쪽 회전
        img = rotate_left(img)
        # 4) 저장
        out_path = os.path.join(out_dir, f"{name}.nii.gz")
        nib.save(img, out_path)
        print(f"[{case_id}] {name} → {out_path}")

def find_series_dir(case_dir, keyword):
    # BTP: 하위 폴더 중 keyword 가 포함된 폴더 이름을 찾아 반환
    for d in os.listdir(case_dir):
        if keyword.lower() in d.lower():
            return os.path.join(case_dir, d)
    raise FileNotFoundError(f"No series folder containing '{keyword}' in {case_dir}")

def preprocess_btp(case_id, args):
    in_dir  = os.path.join(args.input, case_id)
    out_dir = os.path.join(args.output, case_id)
    os.makedirs(out_dir, exist_ok=True)

    # BTP 에서 뽑아낼 시퀀스 (Brats 에 맞춘 5개만)
    seq_map = {
        "flair":   "FLAIRreg",
        "t1pre":   "T1prereg",
        "t1ce":    "T1post",      # Brats 의 t1ce 에 대응
        "t2":      "T2reg",
        "mask":    "MaskTumor",
    }

    for name, keyword in seq_map.items():
        try:
            series_dir = find_series_dir(in_dir, keyword)
        except FileNotFoundError:
            print(f"[{case_id}] *SKIP* no '{keyword}' series")
            continue
        # DICOM 시리즈 → numpy volume
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(series_dir)
        reader.SetFileNames(files)
        img_sitk = reader.Execute()
        arr = sitk.GetArrayFromImage(img_sitk)  # (z,y,x)

        # 1) Resample to 1mm
        spacing_in = img_sitk.GetSpacing()[::-1]  # SITK: (x,y,z)
        factors = [s_out/s_in for s_in,s_out in zip(spacing_in, args.spacing)]
        arr = zoom(arr, factors, order=1)

        # 2) Normalize (mask 제외)
        if name != "mask":
            m, std = arr.mean(), arr.std()
            arr = (arr - m) / (std + 1e-8)

        # 3) Orientation 보정 (Brats 와 일치시키기 위해)
        arr = np.rot90(arr, k=1, axes=(2,1))  # DICOM (z,y,x): rotate axial

        # 4) NIfTI 로 저장
        affine = np.diag(list(args.spacing) + [1])
        img_out = nib.Nifti1Image(arr, affine)
        out_path = os.path.join(out_dir, f"{name}.nii.gz")
        nib.save(img_out, out_path)
        print(f"[{case_id}] {name} → {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  choices=["brats21","btp"], required=True)
    p.add_argument("--input",    type=str, required=True)
    p.add_argument("--output",   type=str, required=True)
    p.add_argument("--spacing",  type=float, nargs=3, default=[1,1,1])
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.dataset == "brats21":
        cases = sorted([d for d in os.listdir(args.input) if d.startswith("BraTS2021")])
        print(f"Found {len(cases)} Brats21 cases: {cases[:5]} …")
        for c in cases:
            preprocess_brats(c, args)

    else:  # btp
        cases = sorted([d for d in os.listdir(args.input) if d.startswith("PGBM-")])
        print(f"Found {len(cases)} BTP cases: {cases[:5]} …")
        for c in cases:
            preprocess_btp(c, args)

if __name__ == "__main__":
    main()
