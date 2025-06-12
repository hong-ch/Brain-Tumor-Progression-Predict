import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from torchvision import models
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class BrainTumor2DSliceDataset(Dataset):
    """환자 ID 기반 2D 슬라이스 데이터셋 클래스 (마스크 크기 예측)"""
    def __init__(self, root_dir, csv_path, use_single_timepoint=True):
        self.root_dir = root_dir
        self.use_single_timepoint = use_single_timepoint
        try:
            self.meta_df = pd.read_csv(csv_path)
            self.meta_df['Patient ID'] = self.meta_df['Patient ID'].astype(str)
        except Exception:
            self.meta_df = pd.DataFrame()
            return
        self.sequences = []
        self.patient_ids = []
        self._build_sequences()

    def _validate_folder(self, path):
        if not os.path.exists(path):
            return False
        patterns = [
            ['t1ce.nii.gz', 'mask.nii.gz'],
            ['flair.nii.gz', 'mask.nii.gz'],
            ['t1.nii.gz', 'mask.nii.gz'],
            ['t2.nii.gz', 'mask.nii.gz'],
        ]
        for pattern in patterns:
            if all(os.path.exists(os.path.join(path, f)) and 
                   os.path.getsize(os.path.join(path, f)) >= 1024 
                   for f in pattern):
                return True
        return False

    def _get_file_pattern(self, path):
        patterns = [
            ['t1ce.nii.gz', 'mask.nii.gz'],
            ['flair.nii.gz', 'mask.nii.gz'],
            ['t1.nii.gz', 'mask.nii.gz'],
            ['t2.nii.gz', 'mask.nii.gz'],
        ]
        for pattern in patterns:
            if all(os.path.exists(os.path.join(path, f)) for f in pattern):
                return pattern
        return patterns[0]

    def _get_valid_timepoints(self, patient_path):
        if not os.path.exists(patient_path):
            return []
        valid_timepoints = []
        try:
            for date_folder in sorted(os.listdir(patient_path)):
                date_path = os.path.join(patient_path, date_folder)
                if os.path.isdir(date_path) and self._validate_folder(date_path):
                    valid_timepoints.append({'date': date_folder, 'path': date_path})
        except:
            pass
        return valid_timepoints

    def _build_sequences(self):
        for _, row in self.meta_df.iterrows():
            patient_id = str(row['Patient ID'])
            patient_path = os.path.join(self.root_dir, patient_id)
            valid_timepoints = self._get_valid_timepoints(patient_path)
            if not valid_timepoints:
                continue
            if len(valid_timepoints) == 1 and self.use_single_timepoint:
                mask_path = os.path.join(valid_timepoints[0]['path'], 'mask.nii.gz')
                mask_vol = nib.load(mask_path).get_fdata()
                for slice_idx in range(22):
                    mask_slice = mask_vol[slice_idx, :, :]
                    mask_sum = np.sum(mask_slice > 0)
                    if mask_sum == 0:
                        continue  # 마스크 크기가 0이면 건너뜀
                    self.sequences.append({
                        'input_path': valid_timepoints[0]['path'],
                        'target_path': valid_timepoints[0]['path'],
                        'patient_id': patient_id,
                        'slice_idx': slice_idx,
                        'patient_info': row.to_dict(),
                        'is_single_timepoint': True
                    })
                    self.patient_ids.append(patient_id)
            elif len(valid_timepoints) > 1:
                for i in range(len(valid_timepoints) - 1):
                    input_mask_path = os.path.join(valid_timepoints[i]['path'], 'mask.nii.gz')
                    target_mask_path = os.path.join(valid_timepoints[i+1]['path'], 'mask.nii.gz')
                    input_mask_vol = nib.load(input_mask_path).get_fdata()
                    target_mask_vol = nib.load(target_mask_path).get_fdata()
                    for slice_idx in range(22):
                        input_mask_slice = input_mask_vol[slice_idx, :, :]
                        target_mask_slice = target_mask_vol[slice_idx, :, :]
                        input_sum = np.sum(input_mask_slice > 0)
                        target_sum = np.sum(target_mask_slice > 0)
                        if input_sum == 0 and target_sum == 0:
                            continue  # 두 시점 모두 마스크 크기가 0이면 추가하지 않음
                        self.sequences.append({
                            'input_path': valid_timepoints[i]['path'],
                            'target_path': valid_timepoints[i+1]['path'],
                            'patient_id': patient_id,
                            'slice_idx': slice_idx,
                            'patient_info': row.to_dict(),
                            'is_single_timepoint': False
                        })
                        self.patient_ids.append(patient_id)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        pattern = self._get_file_pattern(seq['input_path'])
        input_vol = nib.load(os.path.join(seq['input_path'], pattern[0])).get_fdata()
        input_mask = nib.load(os.path.join(seq['input_path'], pattern[1])).get_fdata()
        target_mask = nib.load(os.path.join(seq['target_path'], pattern[1])).get_fdata()
        slice_idx = seq['slice_idx']
        input_data = torch.stack([
            torch.FloatTensor(input_vol[slice_idx, :, :]),
            torch.FloatTensor(self._preprocess_mask(input_mask[slice_idx, :, :]))
        ])
        target_mask_slice = self._preprocess_mask(target_mask[slice_idx, :, :])
        target_mask_size = torch.sum(torch.FloatTensor(target_mask_slice)).item()
        return input_data, torch.FloatTensor([target_mask_size])

    def _preprocess_mask(self, mask_slice):
        return (np.nan_to_num(mask_slice, nan=0.0) > 0).astype(np.float32)

class TumorGrowthPredictor2D(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            weight_mean = original_conv1.weight.mean(dim=1, keepdim=True)
            self.resnet.conv1.weight[:, 0:1, :, :] = weight_mean
            self.resnet.conv1.weight[:, 1:2, :, :] = weight_mean
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.resnet(x)

def train_model(data_dir, csv_path, epochs=100, batch_size=1, learning_rate=1e-4, 
                use_single_timepoint=True, train_size=0.75, random_state=42, print_every=10):
    full_dataset = BrainTumor2DSliceDataset(data_dir, csv_path, use_single_timepoint)
    unique_patients = sorted(list(set(full_dataset.patient_ids)))
    gss_test = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(unique_patients, groups=unique_patients))
    train_val_patients = [unique_patients[i] for i in train_val_idx]
    test_patients = [unique_patients[i] for i in test_idx]
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_patients, groups=train_val_patients))
    train_patients = [train_val_patients[i] for i in train_idx]
    val_patients = [train_val_patients[i] for i in val_idx]
    print(f"▶ Train 환자({len(train_patients)}명): {sorted(train_patients)}")
    print(f"▶ Validation 환자({len(val_patients)}명): {sorted(val_patients)}")
    print(f"▶ Test 환자({len(test_patients)}명): {sorted(test_patients)}")
    train_indices = [i for i, pid in enumerate(full_dataset.patient_ids) if pid in train_patients]
    val_indices = [i for i, pid in enumerate(full_dataset.patient_ids) if pid in val_patients]
    test_indices = [i for i, pid in enumerate(full_dataset.patient_ids) if pid in test_patients]
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices) if val_indices else None
    test_dataset = Subset(full_dataset, test_indices) if test_indices else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TumorGrowthPredictor2D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_batches = 0
        for batch_idx, (inputs, target_sizes) in enumerate(train_loader):
            try:
                inputs, target_sizes = inputs.to(device), target_sizes.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), target_sizes.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                total_batches += 1
                if (batch_idx + 1) % print_every == 0:
                    print(f'Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | Batch Loss: {loss.item():.6f}')
            except Exception as e:
                print(f"배치 처리 오류: {e}")
                continue
        avg_train_loss = train_loss / total_batches if total_batches > 0 else float('inf')
        print(f'\nEpoch [{epoch+1}/{epochs}] | Average Train Loss: {avg_train_loss:.6f}\n')
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_idx, (inputs, target_sizes) in enumerate(val_loader):
                    try:
                        inputs, target_sizes = inputs.to(device), target_sizes.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs.squeeze(), target_sizes.squeeze())
                        val_loss += loss.item()
                        val_batches += 1
                        if (batch_idx + 1) % print_every == 0:
                            print(f'Epoch [{epoch+1}/{epochs}] Val Batch [{batch_idx+1}/{len(val_loader)}] | Val Batch Loss: {loss.item():.6f}')
                    except Exception as e:
                        print(f"검증 배치 처리 오류: {e}")
                        continue
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            print(f'Epoch [{epoch+1}/{epochs}] | Average Val Loss: {avg_val_loss:.6f}\n')
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_tumor_size_model.pth')
                print(f"새로운 최고 모델 저장됨 (Loss: {best_loss:.6f})")
    return model, test_loader

def test_model(model, test_loader, device, print_every=10):
    print("\n=== 테스트 시작 ===")
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    total_batches = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target_sizes) in enumerate(test_loader):
            try:
                inputs, target_sizes = inputs.to(device), target_sizes.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), target_sizes.squeeze())
                test_loss += loss.item()
                total_batches += 1
                predictions.append(outputs.squeeze().item())
                targets.append(target_sizes.squeeze().item())
                if (batch_idx + 1) % print_every == 0:
                    print(f'Test Batch [{batch_idx+1}/{len(test_loader)}] | Batch Loss: {loss.item():.6f}')
            except Exception as e:
                print(f"테스트 배치 처리 오류: {e}")
                continue
    if total_batches > 0:
        avg_test_loss = test_loss / total_batches
        print(f'\n=== 테스트 결과 ===')
        print(f'평균 테스트 손실 (MSE): {avg_test_loss:.6f}')
        print(f'처리된 배치 수: {total_batches}')
        predictions = np.array(predictions)
        targets = np.array(targets)
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets)**2)
        print(f'평균 절대 오차 (MAE): {mae:.6f}')
        print(f'평균 제곱 오차 (MSE): {mse:.6f}')
    else:
        print("테스트 배치가 처리되지 않았습니다.")
    return avg_test_loss if total_batches > 0 else float('inf'), predictions, targets

def plot_pred_vs_true(predictions, targets, title='Predicted vs True Tumor Mask Size'):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=targets, y=predictions, alpha=0.6)
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.title(title)
    plt.xlabel('True Tumor Mask Size')
    plt.ylabel('Predicted Tumor Mask Size')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    data_dir = "/mnt/ssd/brain-tumor-prediction/data/btp_reg_flirt_all_fixed22_nostrip"  # 실제 데이터 경로로 수정
    csv_path = "/mnt/ssd/brain-tumor-prediction/csv/btp_meta.csv"  # 실제 메타데이터 CSV 경로로 수정
    if not os.path.exists(data_dir) or not os.path.exists(csv_path):
        print("데이터 경로 오류!")
        return
    model, test_loader = train_model(
        data_dir, 
        csv_path, 
        use_single_timepoint=False,
        train_size=0.75,
        random_state=42,
        print_every=10
    )
    if model and test_loader:
        print("훈련 완료")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists('best_tumor_size_model.pth'):
            model.load_state_dict(torch.load('best_tumor_size_model.pth'))
            print("최고 성능 모델을 로드했습니다.")
        test_loss, predictions, targets = test_model(model, test_loader, device, print_every=10)
        print(f"최종 테스트 완료 - 평균 손실: {test_loss:.6f}")
        plot_pred_vs_true(predictions, targets)
    else:
        print("훈련 실패")

if __name__ == '__main__':
    main()
