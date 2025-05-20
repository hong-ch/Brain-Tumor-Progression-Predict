#!/usr/bin/env python3
import os
import re
import argparse

def rename_sessions(root_dir):
    """
    btp_preproc 루트 아래, 각 PGBM-xxx 디렉토리 내의
    세션 폴더명을 'MM-DD-YYYY' 형식의 날짜만 남기도록 바꿉니다.
    """
    date_re = re.compile(r"^(\d{2}-\d{2}-\d{4})")
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

            new_name = m.group(1)  # ex: "04-02-1992"
            new_path = os.path.join(case_dir, new_name)
            if os.path.abspath(new_path) == os.path.abspath(sess_path):
                # 이미 원하는 이름이면 패스
                continue

            if os.path.exists(new_path):
                print(f"⚠️  대상 폴더 '{new_name}' 이(가) 이미 존재합니다. '{sess}' 건너뜁니다.")
            else:
                os.rename(sess_path, new_path)
                print(f"✔ '{case}/{sess}' → '{case}/{new_name}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Rename BTP session folders to date-only names (MM-DD-YYYY)."
    )
    p.add_argument(
        "--root",
        required=True,
        help="Path to btp_preproc root directory (e.g. data/btp_preproc)"
    )
    args = p.parse_args()

    rename_sessions(args.root)
