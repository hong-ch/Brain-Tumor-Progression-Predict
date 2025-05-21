#!/usr/bin/env bash
set -e

# ── 1) FSL 환경 로드 ──
export FSLDIR=/mnt/ssd/brain-tumor-prediction/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH

# ── 2) 경로 설정 ──
INPUT_ROOT="data/btp_preproc"
OUTPUT_ROOT="data/btp_reg_linear_all"
mkdir -p "$OUTPUT_ROOT"

# ── 3) 케이스별 반복 ──
for case_path in ${INPUT_ROOT}/PGBM-*; do
  case_id=$(basename "$case_path")
  # 날짜순 정렬 (MM-DD-YYYY 포맷)
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

  # 4) BET → FLAIR 만 skull-strip
  bet "$fixed_dir/flair.nii.gz"   "$fixed_out/flair_brain"   -f 0.3 -g 0 -m
  bet "$moving_dir/flair.nii.gz"  "$moving_out/flair_brain"  -f 0.3 -g 0 -m

  FIXED_BRAIN="$fixed_out/flair_brain.nii.gz"
  MOVING_BRAIN="$moving_out/flair_brain.nii.gz"

  # 5) FLIRT → affine 계산 (FLAIR 기준)
  flirt -in  "$MOVING_BRAIN" \
        -ref "$FIXED_BRAIN" \
        -omat "$moving_out/affine.mat" \
        -out  "$moving_out/flair2fixed_affine.nii.gz"

  # 6) 네 개 모달리티에 affine 적용
  for mod in flair t1 t1ce t2; do
    IN_MOV="$moving_dir/${mod}.nii.gz"
    REF_FIX="$fixed_dir/${mod}.nii.gz"
    OUT_REG="$moving_out/${mod}2fixed_affine.nii.gz"

    flirt -in "$IN_MOV" \
          -ref "$REF_FIX" \
          -init "$moving_out/affine.mat" \
          -out "$OUT_REG" \
          -applyxfm \
          -interp trilinear
    echo "    ✔ ${mod} → $(basename $OUT_REG)"
  done

  echo "✅ ${case_id} 모든 시퀀스 선형 정합 완료"
done

echo "▶ 전체 케이스 처리 완료 (결과: $OUTPUT_ROOT)"
