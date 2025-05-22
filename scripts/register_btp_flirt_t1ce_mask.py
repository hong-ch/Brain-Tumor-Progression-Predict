#!/usr/bin/env python3
# scripts/register_btp_flirt_t1ce_mask.py

import os
import argparse
import multiprocessing
from functools import partial
from subprocess import check_call

def register_fsl(moving_image, fixed_image, out_image, mat_path,
                 dof=6, interp="nearestneighbour", threads=1, overwrite=False):
    """
    Run FLIRT rigid registration: moving → fixed.
    """
    cmd = [
        "flirt",
        "-in", moving_image,
        "-ref", fixed_image,
        "-out", out_image,
        "-omat", mat_path,
        "-dof", str(dof),
        "-interp", interp,
        "-searchrx", "-90", "90",
        "-searchry", "-90", "90",
        "-searchrz", "-90", "90",
        "-cost", "corratio",
        "-bins", "256"
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    if not overwrite and os.path.exists(out_image) and os.path.getsize(out_image) > 0:
        return
    check_call(cmd, env=env)

def apply_mat(moving_image, reference, mat_path, out_image,
              interp="nearestneighbour", threads=1, overwrite=False):
    """
    Apply existing affine (.mat) to moving_image → reference space.
    """
    cmd = [
        "flirt",
        "-in", moving_image,
        "-ref", reference,
        "-applyxfm",
        "-init", mat_path,
        "-out", out_image,
        "-interp", interp,
        "-paddingsize", "0.0"
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    if not overwrite and os.path.exists(out_image) and os.path.getsize(out_image) > 0:
        return
    check_call(cmd, env=env)

def process_case(case_path, fixed_sess, moving_sess, input_root, output_root, threads):
    case_id = os.path.basename(case_path)
    fixed_dir  = os.path.join(input_root,  case_id, fixed_sess)
    moving_dir = os.path.join(input_root,  case_id, moving_sess)
    fixed_out  = os.path.join(output_root, case_id, fixed_sess)
    moving_out = os.path.join(output_root, case_id, moving_sess)
    os.makedirs(fixed_out,  exist_ok=True)
    os.makedirs(moving_out, exist_ok=True)

    print(f"▶ {case_id}: [fixed: {fixed_sess}] ← [moving: {moving_sess}]")

    FIXED_IMG   = os.path.join(fixed_out,  "t1ce2fixed_affine.nii.gz")  # dummy, won't be created
    MOVING_IMG  = os.path.join(moving_out, "t1ce2fixed_affine.nii.gz")
    FIXED_REF   = os.path.join(fixed_dir,  "t1ce_brain.nii.gz")
    MOVING_IN   = os.path.join(moving_dir, "t1ce_brain.nii.gz")
    MAT_PATH    = os.path.join(moving_out, "t1ce2fixed_affine.mat")

    # 1) skull‐stripped T1CE rigid registration
    register_fsl(
        moving_image = MOVING_IN,
        fixed_image  = FIXED_REF,
        out_image    = MOVING_IMG,
        mat_path     = MAT_PATH,
        dof          = 6,
        interp       = "trilinear",
        threads      = threads
    )

    # 2) tumor mask registration (nearest neighbour)
    FIXED_MASK  = os.path.join(fixed_dir,  "mask.nii.gz")
    MOVING_MASK = os.path.join(moving_dir, "mask.nii.gz")
    OUT_MASK    = os.path.join(moving_out, "mask2fixed_affine.nii.gz")

    apply_mat(
        moving_image = MOVING_MASK,
        reference    = FIXED_MASK,
        mat_path     = MAT_PATH,
        out_image    = OUT_MASK,
        interp       = "nearestneighbour",
        threads      = threads
    )

    print(f"✅ {case_id} registration complete\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root",  required=True,
                   help="preproc_fixed22 root (skull-strip+t1ce_brain.nii.gz)")
    p.add_argument("--output_root", required=True,
                   help="where to write registered (FLIRT) outputs")
    p.add_argument("--threads",     type=int, default=1,
                   help="OMP_NUM_THREADS for FLIRT")
    args = p.parse_args()

    # make sure fsl is on PATH
    if "FSLDIR" not in os.environ:
        raise RuntimeError("Please source your FSL setup (FSLDIR/etc/fslconf/fsl.sh) before running")

    # iterate cases
    for case in sorted(os.listdir(args.input_root)):
        case_path = os.path.join(args.input_root, case)
        if not os.path.isdir(case_path) or not case.startswith("PGBM-"):
            continue

        # find exactly two session‐folders, sort them lexically (YYYY-MM-DD)
        sess = sorted(d for d in os.listdir(case_path)
                      if os.path.isdir(os.path.join(case_path, d)))
        if len(sess) != 2:
            print(f"⚠️  {case}: found {len(sess)} sessions, skipping")
            continue

        fixed_sess, moving_sess = sess
        process_case(case_path, fixed_sess, moving_sess,
                     args.input_root, args.output_root, args.threads)

if __name__ == "__main__":
    main()
