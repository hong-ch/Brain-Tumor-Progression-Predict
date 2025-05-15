#!/usr/bin/env python3
# scripts/preprocess.py

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import SimpleITK as sitk

def resample_to_spacing(img, target_spacing=(1,1,1)):
    """NIfTI 이미지(img) → 지정한 spacing 으로 재샘플링된 NIfTI 반환"""
    data = img.get_fdata()
    orig_spacing = img.header.get_zooms()[:3]
    factors = [o/t for o,t in zip(orig_spacing, target_spacing)]
    resampled = zoom(data, factors, order=1)
    new_affine = img.affine.copy()
    new_affine[:3, :3] = np.diag(target_spacing)
    return nib.Nifti1Image(resampled, new_affine)

def zscore_normalize(x):
    """
    - 만약 x가 NumPy ndarray 면: (x - mean)/std 반환 (ndarray)
    - 만약 x가 Nifti1Image 면: 이미지 데이터 z-score 정규화 후 Nifti1Image 반환
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
    """하위 폴더 전체 탐색해서 keyword 포함된 첫 디렉토리 경로 반환"""
    for r, dirs, _ in os.walk(root):
        for d in dirs:
            if keyword.lower() in d.lower():
                return os.path.join(r, d)
    raise FileNotFoundError(f"No series containing '{keyword}' under {root}")

def preprocess_btp(case_id, args):
    in_root  = os.path.join(args.input,  case_id)
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
        in_day  = os.path.join(in_root, day)
        if not os.path.isdir(in_day): continue

        out_day = os.path.join(out_root, day)
        os.makedirs(out_day, exist_ok=True)
        print(f"[{case_id}] 날짜: {day}")

        for name, key in seq_map.items():
            try:
                series_dir = find_series_dir(in_day, key)
            except FileNotFoundError:
                print(f"  - SKIP: '{key}' 없음")
                continue

            # 1) DICOM → (Z,Y,X) NumPy 배열
            reader = sitk.ImageSeriesReader()
            files  = reader.GetGDCMSeriesFileNames(series_dir)
            reader.SetFileNames(files)
            img_si = reader.Execute()
            arr_z_y_x = sitk.GetArrayFromImage(img_si)

            # 2) spacing 보정 (zoom)
            sp_in = img_si.GetSpacing()[::-1]  # SITK: (x,y,z) → (z,y,x)
            factors = [si/so for si,so in zip(sp_in, args.spacing)]
            arr_z_y_x = zoom(arr_z_y_x, factors, order=1)

            # 3) normalize (mask 제외)
            if name != "mask":
                arr_z_y_x = zscore_normalize(arr_z_y_x)  # ndarray 지원!

            # 4) 축 변환 → (X,Y,Z)
            arr_x_y_z = arr_z_y_x.transpose(2,1,0)

            # 5) NIfTI 저장
            affine = np.diag(list(args.spacing)+[1])
            img_out = nib.Nifti1Image(arr_x_y_z, affine)
            out_path = os.path.join(out_day, f"{name}.nii.gz")
            nib.save(img_out, out_path)
            print(f"    ✔ {name} 저장, shape={arr_x_y_z.shape}")

def preprocess_brats(case_id, args):
    in_dir  = os.path.join(args.input, case_id)
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
        # 1) spacing 보정
        img = resample_to_spacing(img, args.spacing)
        # 2) normalize (mask 제외)
        if name != "mask":
            img = zscore_normalize(img)            # Nifti1Image 지원!
        # 3) 저장
        out_path = os.path.join(out_dir, f"{name}.nii.gz")
        nib.save(img, out_path)
        print(f"[BRATS] {case_id}/{name} → {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["btp","brats21"], required=True)
    p.add_argument("--input",   required=True)
    p.add_argument("--output",  required=True)
    p.add_argument("--spacing", nargs=3, type=float, default=[1,1,1])
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for case in sorted(os.listdir(args.input)):
        if args.dataset=="btp"     and case.startswith("PGBM-"):
            preprocess_btp(case, args)
        if args.dataset=="brats21" and case.startswith("BraTS2021"):
            preprocess_brats(case, args)

if __name__=="__main__":
    main()
