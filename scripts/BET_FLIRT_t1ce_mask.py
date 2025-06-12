#!/usr/bin/env python3
import os
from nipype.interfaces.fsl import BET, FLIRT

def skull_strip(in_file, out_file, frac=0.3, vertical_gradient=0.0):
    """
    BET으로 skull‐strip 수행
    - in_file:  원본 NIfTI 경로
    - out_file: skull‐stripped NIfTI 경로 (mask는 자동으로 out_file + '_mask')
    """
    bet = BET()
    bet.inputs.in_file  = in_file
    bet.inputs.out_file = out_file
    bet.inputs.mask     = True
    bet.inputs.frac     = frac
    bet.inputs.vertical_gradient = vertical_gradient
    res = bet.run()  # 실행
    return res.outputs.out_file, res.outputs.mask_file

def affine_register(moving, reference, out_file, out_mat, dof=12, interp='trilinear'):
    """
    FLIRT으로 affine 정합 수행
    - moving:    이동 이미지 경로
    - reference: 기준 이미지 경로
    - out_file:  정합된 출력 이미지 경로
    - out_mat:   생성할 affine 매트릭스(.mat) 경로
    """
    flt = FLIRT()
    flt.inputs.in_file          = moving
    flt.inputs.reference        = reference
    flt.inputs.out_file         = out_file
    flt.inputs.out_matrix_file  = out_mat
    flt.inputs.dof              = dof
    flt.inputs.interp           = interp
    flt.inputs.cost             = 'corratio'
    flt.inputs.searchr_x        = [-90, 90]
    flt.inputs.searchr_y        = [-90, 90]
    flt.inputs.searchr_z        = [-90, 90]
    res = flt.run()
    return res.outputs.out_file, res.outputs.out_matrix_file

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="BET → FLIRT pipeline using NiPype FSL interfaces"
    )
    p.add_argument("--input_root",  required=True,
                   help="원본 DICOM→NIfTI 전처리 폴더 (예: data/btp_preproc_fixed22)")
    p.add_argument("--output_root", required=True,
                   help="정합 결과를 저장할 루트 폴더")
    p.add_argument("--frac", type=float, default=0.3,
                   help="BET skull‐strip 강도 (–f)")
    p.add_argument("--grad", type=float, default=0.0,
                   help="BET vertical gradient (–g)")
    args = p.parse_args()

    # ─── 케이스별/세션별 반복 ─────────────────
    for case_path in sorted(os.listdir(args.input_root)):
        case_dir = os.path.join(args.input_root, case_path)
        if not os.path.isdir(case_dir) or not case_path.startswith("PGBM-"):
            continue

        sessions = sorted(d for d in os.listdir(case_dir)
                          if os.path.isdir(os.path.join(case_dir, d)))
        if len(sessions) != 2:
            print(f"⚠️  {case_path}: 세션이 2개가 아님, 건너뜁니다.")
            continue

        fixed_dt, moving_dt = sessions
        fixed_dir  = os.path.join(case_dir, fixed_dt)
        moving_dir = os.path.join(case_dir, moving_dt)

        out_case = os.path.join(args.output_root, case_path)
        out_fixed  = os.path.join(out_case, fixed_dt)
        out_moving = os.path.join(out_case, moving_dt)
        os.makedirs(out_fixed,  exist_ok=True)
        os.makedirs(out_moving, exist_ok=True)

        print(f"▶ {case_path}: [Fixed: {fixed_dt}] [Moving: {moving_dt}]")

        # 1) skull‐strip both sessions (T1CE만 예시; 필요 시 다른 시퀀스도 반복)
        in_fixed_t1ce  = os.path.join(fixed_dir,  "t1ce.nii.gz")
        in_moving_t1ce = os.path.join(moving_dir, "t1ce.nii.gz")

        out_fixed_brain,  out_fixed_mask  = skull_strip(
            in_fixed_t1ce,
            os.path.join(out_fixed,  "t1ce_brain.nii.gz"),
            frac=args.frac, vertical_gradient=args.grad
        )
        out_moving_brain, out_moving_mask = skull_strip(
            in_moving_t1ce,
            os.path.join(out_moving, "t1ce_brain.nii.gz"),
            frac=args.frac, vertical_gradient=args.grad
        )

        print(f"    ✔ BET Fixed brain:  {out_fixed_brain}")
        print(f"    ✔ BET Fixed mask:   {out_fixed_mask}")
        print(f"    ✔ BET Moving brain: {out_moving_brain}")
        print(f"    ✔ BET Moving mask:  {out_moving_mask}")

        # 2) affine 정합: moving_brain → fixed_brain
        reg_img, reg_mat = affine_register(
            moving = out_moving_brain,
            reference = out_fixed_brain,
            out_file  = os.path.join(out_moving, "t1ce2fixed_affine.nii.gz"),
            out_mat   = os.path.join(out_moving, "affine.mat"),
            dof       = 12,
            interp    = 'trilinear'
        )
        print(f"    ✔ FLIRT out_img: {reg_img}")
        print(f"    ✔ FLIRT out_mat: {reg_mat}")

        # 3) mask에도 동일한 transform 적용 (nearest‐neighbour)
        reg_mask_img, _ = affine_register(
            moving     = out_moving_mask,
            reference  = out_fixed_mask,
            out_file   = os.path.join(out_moving, "mask2fixed_affine.nii.gz"),
            out_mat    = reg_mat,
            dof        = 12,
            interp     = 'nearestneighbour'
        )
        print(f"    ✔ FLIRT mask: {reg_mask_img}")

        print(f"✅ {case_path} 처리 완료\n")
