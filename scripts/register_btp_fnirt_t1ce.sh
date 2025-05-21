#!/usr/bin/env bash
set -e

# ── 1) FSL 환경 로드 & 멀티스레드 설정 ──
export FSLDIR=/mnt/ssd/brain-tumor-prediction/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH

export OMP_NUM_THREADS=3  
export FSLPARALLEL=3  

# ── 2) 경로 설정 ──
INPUT_ROOT="data/btp_preproc"
OUTPUT_ROOT="data/btp_reg_fnirt_t1ce"
mkdir -p "$OUTPUT_ROOT"

# ── 3) 케이스별 반복 ──
for case_path in ${INPUT_ROOT}/PGBM-*; do
  case_id=$(basename "$case_path")

  # MM-DD-YYYY 포맷 날짜순 정렬
  sessions=( $(ls "$case_path" | sort -V) )
  if [ "${#sessions[@]}" -ne 2 ]; then
    echo "⚠️  ${case_id}: 세션이 2개가 아닙니다, 건너뜁니다."
    continue
  fi

  fixed_sess=${sessions[0]}
  moving_sess=${sessions[1]}
  fixed_dir="$case_path/$fixed_sess"
  moving_dir="$case_path/$moving_sess"

  fixed_out="$OUTPUT_ROOT/$case_id/$fixed_sess"
  moving_out="$OUTPUT_ROOT/$case_id/$moving_sess"
  mkdir -p "$fixed_out" "$moving_out"

  echo "▶ ${case_id}: [고정: $fixed_sess] ➔ [이동: $moving_sess]"

  # A) BET: T1CE skull‐strip → brain + mask
  bet "$fixed_dir/t1ce.nii.gz"   "$fixed_out/t1ce_brain"   -f 0.3 -g 0 -m
  bet "$moving_dir/t1ce.nii.gz"  "$moving_out/t1ce_brain"  -f 0.3 -g 0 -m

  FIXED_BRAIN="$fixed_out/t1ce_brain.nii.gz"
  FIXED_MASK="$fixed_out/t1ce_brain_mask.nii.gz"
  MOVING_BRAIN="$moving_out/t1ce_brain.nii.gz"
  MOVING_MASK="$moving_out/t1ce_brain_mask.nii.gz"

  # B) FLIRT: 선형 정합
  flirt -in  "$MOVING_BRAIN" \
        -ref "$FIXED_BRAIN" \
        -omat "$moving_out/affine.mat" \
        -out  "$moving_out/t1ce2fixed_affine.nii.gz"

  # C) FNIRT: 비선형 정합 (마스크 활용)
  fnirt --in="$MOVING_BRAIN" \
        --ref="$FIXED_BRAIN" \
        --aff="$moving_out/affine.mat" \
        --inmask="$MOVING_MASK" \
        --refmask="$FIXED_MASK" \
        --cout="$moving_out/warpcoef.nii.gz" \
        --iout="$moving_out/t1ce2fixed_fnirt.nii.gz"

  echo "✅ ${case_id} FNIRT 정합 완료"
done

echo "▶ 모든 케이스 FNIRT 정합 완료 (결과: $OUTPUT_ROOT)"
