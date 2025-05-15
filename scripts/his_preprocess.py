#!/usr/bin/env python3
# scripts/preprocess.py

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import SimpleITK as sitk
from skimage.exposure import match_histograms

# ─── 헬퍼 함수들 ──────────────────────────────────────────

def resample_to_spacing(img, target_spacing):
    data = img.get_fdata()
    orig_sp = img.header.get_zooms()[:3]
    factors = [o/t for o,t in zip(orig_sp, target_spacing)]
    res = zoom(data, factors, order=1)
    new_aff = img.affine.copy()
    new_aff[:3,:3] = np.diag(target_spacing)
    return nib.Nifti1Image(res, new_aff)

def zscore_normalize(img):
    d = img.get_fdata()
    m,s = d.mean(), d.std()
    return nib.Nifti1Image((d-m)/(s+1e-8), img.affine)

def rotate_left(img):
    d = img.get_fdata()
    r = np.rot90(d, k=1, axes=(0,1))
    R = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    return nib.Nifti1Image(r, img.affine @ R)

def find_series_dir(case_dir, keyword):
    """주어진 case_dir(여기선 날짜 폴더) 하위에 keyword 포함된 폴더를 재귀 탐색"""
    for root, dirs, _ in os.walk(case_dir):
        for d in dirs:
            if keyword.lower() in d.lower():
                return os.path.join(root, d)
    raise FileNotFoundError(f"No series folder containing '{keyword}' under {case_dir}")

# ─── Brats21 전처리 ───────────────────────────────────────

def preprocess_brats(case, args):
    in_dir  = os.path.join(args.input, case)
    out_dir = os.path.join(args.output, case)
    os.makedirs(out_dir, exist_ok=True)

    files = {
        "flair": f"{case}_flair.nii.gz",
        "t1":    f"{case}_t1.nii.gz",
        "t1ce":  f"{case}_t1ce.nii.gz",
        "t2":    f"{case}_t2.nii.gz",
        "mask":  f"{case}_seg.nii.gz",
    }

    for name, fname in files.items():
        img = nib.load(os.path.join(in_dir, fname))
        img = resample_to_spacing(img, args.spacing)
        if name != "mask":
            img = zscore_normalize(img)
        img = rotate_left(img)
        nib.save(img, os.path.join(out_dir, f"{name}.nii.gz"))
        print(f"[BRATS] {case}/{name} → saved")

# ─── BTP 전처리 (날짜별 폴더까지 재귀) ─────────────────────

def preprocess_btp(case, args):
    case_dir = os.path.join(args.input, case)
    out_case = os.path.join(args.output, case)

    # Brats21 전처리 결과 중 첫 번째 케이스를 ref로 삼기
    ref_root = args.hist_ref
    ref_case = None
    if ref_root and os.path.isdir(ref_root):
        cands = [d for d in os.listdir(ref_root) if d.startswith("BraTS")]
        if cands:
            ref_case = sorted(cands)[0]
            print(f"[BTP] histogram-ref = {ref_case}")
    ref_template = None
    if ref_case:
        ref_template = os.path.join(ref_root, ref_case, "{mod}.nii.gz")

    seq_map = {
        "flair":  "FLAIRreg",
        "t1":  "T1prereg",
        "t1ce":   "T1post",
        "t2":     "T2reg",
        "mask":   "MaskTumor",
    }

    # case_dir 아래의 모든 날짜 폴더를 순회
    for date in sorted(os.listdir(case_dir)):
        date_dir = os.path.join(case_dir, date)
        if not os.path.isdir(date_dir):
            continue

        out_date = os.path.join(out_case, date)
        os.makedirs(out_date, exist_ok=True)
        print(f"\n[BTP] Processing {case} / {date}")

        for name, key in seq_map.items():
            try:
                series = find_series_dir(date_dir, key)
            except FileNotFoundError:
                print(f"  SKIP {name} (no '{key}')")
                continue

            # DICOM → numpy array
            reader = sitk.ImageSeriesReader()
            fns = reader.GetGDCMSeriesFileNames(series)
            reader.SetFileNames(fns)
            img_si = reader.Execute()
            arr = sitk.GetArrayFromImage(img_si)  # (z,y,x)
            sp_in = img_si.GetSpacing()[::-1]    # (z,y,x)
            fac = [o/t for o,t in zip(sp_in, args.spacing)]
            arr = zoom(arr, fac, order=1)

            # normalize
            if name != "mask":
                arr = (arr - arr.mean())/(arr.std()+1e-8)

                # histogram matching (flair/t1/t1ce/t2 에 모두 적용)
                if ref_template:
                    ref_path = ref_template.format(mod=name)
                    if os.path.exists(ref_path):
                        ref_data = nib.load(ref_path).get_fdata()
                        # multichannel 인자 제거
                        arr = match_histograms(arr, ref_data)
                        print(f"  {name}: hist-matched")
                    else:
                        print(f"  {name}: hist-skip (no ref)")
            # NIfTI 로 저장
            aff = np.diag(list(args.spacing)+[1])
            out_img = nib.Nifti1Image(arr, aff)
            nib.save(out_img, os.path.join(out_date, f"{name}.nii.gz"))
            print(f"  {name} → saved")

# ─── main ────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["brats21","btp"], required=True)
    p.add_argument("--input",   required=True)
    p.add_argument("--output",  required=True)
    p.add_argument("--spacing", nargs=3, type=float, default=[1,1,1])
    p.add_argument("--hist-ref", type=str, default=None,
                   help="Brats21 전처리 폴더 (histogram reference)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for c in sorted(os.listdir(args.input)):
        if args.dataset=="brats21" and c.startswith("BraTS"):
            preprocess_brats(c, args)
        if args.dataset=="btp"      and c.startswith("PGBM-"):
            preprocess_btp(c, args)

if __name__=="__main__":
    main()
