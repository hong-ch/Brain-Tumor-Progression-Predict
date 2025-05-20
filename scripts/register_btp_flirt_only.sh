#!/usr/bin/env bash
set -e

# ── FSL 환경 로드 ──
export FSLDIR=/mnt/ssd/brain-tumor-prediction/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH

# ── 입력/출력 경로 ──
INPUT_ROOT="data/btp_preproc"
OUTPUT_ROOT="data/btp_reg_linear"
mkdir -p "$OUTPUT_ROOT"

for case_path in ${INPUT_ROOT}/PGBM-*; do
  case_id=$(basename "$case_path")
  sessions=( $(ls "$case_path" | sort) )
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

  echo "▶ ${case_id}: [고정:$fixed_sess] ← [이동:$moving_sess]"

  # 1) BET: 뇌만 추출
  bet "$fixed_dir/flair.nii.gz"  "$fixed_out/flair_brain"  -f 0.3 -g 0 -m
  bet "$moving_dir/flair.nii.gz" "$moving_out/flair_brain" -f 0.3 -g 0 -m

  FIXED_BRAIN="$fixed_out/flair_brain.nii.gz"
  MOVING_BRAIN="$moving_out/flair_brain.nii.gz"

  # 2) FLIRT: 선형 정합
  flirt -in  "$MOVING_BRAIN" \
        -ref "$FIXED_BRAIN" \
        -omat "$moving_out/affine.mat" \
        -out  "$moving_out/flair2fixed_affine.nii.gz"

  echo "✅ ${case_id} 선형 정합 완료"
done

echo "▶ 모든 케이스 선형 정합 완료 (결과: $OUTPUT_ROOT)"
