#!/usr/bin/env python3
import os
import re
import argparse

def rename_sessions(root_dir):
    """
    btp_preproc 루트 아래, 각 PGBM-xxx 디렉토리 내의
    세션 폴더명을 'YYYY-MM-DD' 형식의 날짜만 남기도록 바꿉니다.
    """
    # MM-DD-YYYY 형태를 잡아내는 정규식
    date_re = re.compile(r"^(\d{2})-(\d{2})-(\d{4})")
    for case in sorted(os.listdir(root_dir)):
        case_dir = os.path.join(root_dir, case)
        if not os.path.isdir(case_dir):
            continue

        for sess in sorted(os.listdir(case_dir)):
            sess_path = os.path.join(case_dir, sess)
            if not os.path.isdir(sess_path):
                continue

            m = date_re.match(sess)
            if not m:
                print(f"⚠️  '{sess}' 에서 날짜를 찾을 수 없습니다. 건너뜁니다.")
                continue

            mm, dd, yyyy = m.groups()
            new_name = f"{yyyy}-{mm}-{dd}"         # YYYY-MM-DD
            new_path = os.path.join(case_dir, new_name)

            # 이미 같은 이름이면 스킵
            if os.path.abspath(new_path) == os.path.abspath(sess_path):
                continue

            # 충돌 방지: 같은 이름 폴더가 이미 있으면 스킵
            if os.path.exists(new_path):
                print(f"⚠️  대상 폴더 '{new_name}' 이(가) 이미 존재합니다. '{sess}' 건너뜁니다.")
            else:
                os.rename(sess_path, new_path)
                print(f"✔ '{case}/{sess}' → '{case}/{new_name}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Rename BTP session folders to date-only names (YYYY-MM-DD)."
    )
    p.add_argument(
        "--root",
        required=True,
        help="Path to btp_preproc root directory (e.g. data/btp_preproc)"
    )
    args = p.parse_args()

    rename_sessions(args.root)
