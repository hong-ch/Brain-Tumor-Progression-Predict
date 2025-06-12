#!/usr/bin/env bash
set -e

# ── 1) FSL 환경 로드 & 멀티스레드 설정 ──
export FSLDIR=/mnt/ssd/brain-tumor-prediction/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH
export OMP_NUM_THREADS=3
export FSLPARALLEL=3

# ── 2) 경로 설정 ──
INPUT_ROOT="data/btp_preproc_224"
OUTPUT_ROOT="data/btp_reg_flirt_all_224_nostrip"
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

  # 4) T1CE 기준 affine 계산 (이 단계만 fixed→moving 참조)
  flirt -in  "$moving_dir/t1ce.nii.gz" \
        -ref "$fixed_dir/t1ce.nii.gz" \
        -omat "$moving_out/affine.mat" \
        -out  "$moving_out/t1ce2fixed_affine.nii.gz"
  echo "    ✔ T1CE affine matrix 및 샘플 출력 완료"

  # 5) 4개 시퀀스 (flair, t1, t1ce, t2) + mask 에 동일한 affine 적용
  for mod in flair t1 t1ce t2 mask; do
    in_img="$moving_dir/${mod}.nii.gz"
    ref_img="$fixed_dir/${mod}.nii.gz"     # 참조용
    out_img="$moving_out/${mod}2fixed_affine.nii.gz"

    # mask 만 nearest‐neighbour, 나머지는 trilinear
    interp_method=trilinear
    if [ "$mod" == "mask" ]; then
      interp_method=nearestneighbour
    fi

    flirt -in   "$in_img" \
          -ref  "$ref_img" \
          -init "$moving_out/affine.mat" \
          -applyxfm \
          -interp $interp_method \
          -out  "$out_img"

    echo "    ✔ ${mod^^} registered: $(basename $out_img)"
  done

  echo "✅ ${case_id} 모든 시퀀스 선형 정합 완료 (내부 정보는 이동 세션 그대로)"
done

echo "▶ 전체 케이스 처리 완료 (결과 폴더: $OUTPUT_ROOT)"
