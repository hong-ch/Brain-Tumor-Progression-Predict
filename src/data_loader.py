import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def calculate_volume_from_mask(mask_nii):
    """NIfTI 마스크 파일로부터 종양의 부피(mm^3)를 계산합니다. (기존과 동일)"""
    mask_data = mask_nii.get_fdata()
    voxel_dims = mask_nii.header.get_zooms()
    voxel_volume = np.prod(voxel_dims)
    tumor_voxel_count = np.sum(mask_data > 0)
    return tumor_voxel_count * voxel_volume

class BrainTumorDataset(Dataset):
    """뇌종양 예측을 위한 PyTorch 커스텀 데이터셋 클래스 (구조 변경 대응)"""
    def __init__(self, data_dir, patient_ids):
        self.patient_ids = patient_ids
        self.data_files = self._prepare_data_files(data_dir, patient_ids)

    def _prepare_data_files(self, data_dir, patient_ids):
        print(f"Preparing file list for {len(patient_ids)} patients...")
        files = []
        for patient_id in tqdm(patient_ids):
            patient_folder = os.path.join(data_dir, patient_id)
            
            # 환자 폴더 내의 두 시점(날짜) 폴더를 찾습니다.
            session_folders = sorted([d for d in os.listdir(patient_folder) if os.path.isdir(os.path.join(patient_folder, d))])
            
            # 시점 폴더가 2개가 아니면 경고를 출력하고 건너뜁니다.
            if len(session_folders) != 2:
                print(f"Warning: Patient {patient_id} does not have 2 session folders. Skipping.")
                continue
            
            # 시간순으로 정렬되었으므로, 첫 번째가 이전 시점, 두 번째가 이후 시점입니다.
            earlier_session_path = os.path.join(patient_folder, session_folders[0])
            later_session_path = os.path.join(patient_folder, session_folders[1])
            
            # 입력(X)과 타겟(y) 파일 경로를 정확히 지정합니다.
            input_t1ce_path = os.path.join(earlier_session_path, 't1ce.nii.gz')
            target_mask_path = os.path.join(later_session_path, 'mask2fixed_affine.nii.gz')

            # 파일이 실제로 존재하는지 확인합니다.
            if os.path.exists(input_t1ce_path) and os.path.exists(target_mask_path):
                files.append({
                    "input_t1ce_path": input_t1ce_path,
                    "target_mask_path": target_mask_path
                })
            else:
                print(f"Warning: Files not found for patient {patient_id}. Skipping.")

        return files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # 준비된 파일 경로 가져오기
        file_paths = self.data_files[idx]
        
        # .nii.gz 파일 로드
        t1ce_nii = nib.load(file_paths['input_t1ce_path'])
        mask_nii = nib.load(file_paths['target_mask_path'])
        
        # 이미지 데이터 전처리
        t1ce_data = t1ce_nii.get_fdata().astype(np.float32)
        t1ce_data = (t1ce_data - np.min(t1ce_data)) / (np.max(t1ce_data) - np.min(t1ce_data) + 1e-6)
        
        # PyTorch 형식에 맞게 축 변경: (H, W, D) -> (D, H, W) -> (D, C, H, W)
        t1ce_data = np.transpose(t1ce_data, (2, 0, 1))
        t1ce_data = np.expand_dims(t1ce_data, axis=1)
        
        # 타겟 부피 계산
        volume = calculate_volume_from_mask(mask_nii)

        # Numpy 배열을 PyTorch 텐서로 변환
        image_tensor = torch.from_numpy(t1ce_data).float()
        label_tensor = torch.tensor(volume, dtype=torch.float32)

        return image_tensor, label_tensor