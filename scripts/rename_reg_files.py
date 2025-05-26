import os

# 최상위 경로 설정
base_dir = '/mnt/ssd/brain-tumor-prediction/data/btp_reg_flirt_all_fixed22_nostrip'

# PGBM-001 ~ PGBM-020 반복
for patient_folder in sorted(os.listdir(base_dir)):
    patient_path = os.path.join(base_dir, patient_folder)
    if not os.path.isdir(patient_path) or not patient_folder.startswith("PGBM-"):
        continue

    # 날짜 폴더 탐색
    for date_folder in os.listdir(patient_path):
        date_path = os.path.join(patient_path, date_folder)
        if not os.path.isdir(date_path):
            continue

        # 해당 날짜 폴더 내 파일들 처리
        for filename in os.listdir(date_path):
            if '2fixed_affine' in filename:
                old_path = os.path.join(date_path, filename)
                new_filename = filename.replace('2fixed_affine', '')
                new_path = os.path.join(date_path, new_filename)
                
                # 이름 바꾸기
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
