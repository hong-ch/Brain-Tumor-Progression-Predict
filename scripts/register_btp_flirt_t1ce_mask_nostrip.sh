#!/usr/bin/env bash
set -e

# ── 1) FSL 환경 로드 & 멀티스레드 설정 ──
export FSLDIR=/mnt/ssd/brain-tumor-prediction/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH
export OMP_NUM_THREADS=3
export FSLPARALLEL=3

# ── 2) 경로 설정 ──
INPUT_ROOT="data/btp_preproc_fixed22"
OUTPUT_ROOT="data/btp_reg_flirt_t1ce_fixed22_nostrip"
mkdir -p "$OUTPUT_ROOT"

# ── 3) 케이스별 반복 ──
for case_path in ${INPUT_ROOT}/PGBM-*; do
  case_id=$(basename "$case_path")
  sessions=( $(ls "$case_path" | sort -V) )
  if [ "${#sessions[@]}" -ne 2 ]; then
    echo "⚠️  ${case_id}: 세션 수 ≠ 2, 건너뜁니다."
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

  # 4) FLIRT: 원본 T1CE → 원본 T1CE affine 정합
  flirt -in  "$moving_dir/t1ce.nii.gz" \
        -ref "$fixed_dir/t1ce.nii.gz" \
        -omat "$moving_out/affine.mat" \
        -out  "$moving_out/t1ce2fixed_affine.nii.gz"

  # 5) FLIRT: 종양 마스크도 같은 affine 적용 (nearest‐neighbour)
  flirt -in  "$moving_dir/mask.nii.gz" \
        -ref "$fixed_dir/mask.nii.gz" \
        -init "$moving_out/affine.mat" \
        -applyxfm \
        -interp nearestneighbour \
        -out  "$moving_out/mask2fixed_affine.nii.gz"

  echo "✅ ${case_id} T1CE + mask 선형 정합 완료 (skull-strip 제외)"
done

echo "▶ 모든 케이스 T1CE 선형 정합(+mask) 완료 (결과: $OUTPUT_ROOT)"
