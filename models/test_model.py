import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
from scipy import ndimage

class GliomaVolumeDataset(Dataset):
    """
    뇌종양 MRI 데이터셋 클래스
    반응-확산 모델 기반 합성 데이터 로딩 지원
    """
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # MRI 이미지 로딩 (T1, T1Gd, T2, FLAIR)
        patient_id = self.data_info.iloc[idx]['patient_id']
        timepoint = self.data_info.iloc[idx]['timepoint']

        # 다중 모달리티 MRI 로딩
        mri_path = os.path.join(self.data_dir, f"{patient_id}_{timepoint}.npy")
        mri_data = np.load(mri_path)  # Shape: (C, H, W, D)

        # 종양 윤곽선 추출 (임계값 기반)
        contour_1 = self.extract_contour(mri_data, threshold=0.8)  # 강화 코어
        contour_2 = self.extract_contour(mri_data, threshold=0.3)  # 부종 영역

        # 미래 시점 종양 부피 (ground truth)
        future_volume = self.data_info.iloc[idx]['future_volume']

        # 확산성과 증식 매개변수
        diffusivity = self.data_info.iloc[idx]['diffusivity']
        proliferation = self.data_info.iloc[idx]['proliferation']

        sample = {
            'mri': torch.FloatTensor(mri_data),
            'contour_1': torch.FloatTensor(contour_1),
            'contour_2': torch.FloatTensor(contour_2),
            'future_volume': torch.FloatTensor([future_volume]),
            'diffusivity': torch.FloatTensor([diffusivity]),
            'proliferation': torch.FloatTensor([proliferation])
        }

        return sample

    def extract_contour(self, mri_data, threshold):
        """
        임계값 기반 종양 윤곽선 추출
        """
        # 마지막 채널이 세그멘테이션이라고 가정
        segmentation = mri_data[-1]
        contour = (segmentation > threshold).astype(np.float32)
        return contour

class ReactionDiffusionNet(nn.Module):
    """
    반응-확산 뇌종양 성장 모델링을 위한 3D CNN
    두 개의 주요 작업: 세포 밀도 재구성 + 매개변수 추정
    """
    def __init__(self, input_channels=4, num_classes=1):
        super(ReactionDiffusionNet, self).__init__()

        # 인코더 (특징 추출)
        self.encoder = nn.Sequential(
            # 첫 번째 블록
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # 두 번째 블록
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # 세 번째 블록
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        # 세포 밀도 재구성을 위한 디코더
        self.density_decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 매개변수 추정을 위한 회귀 헤드
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.parameter_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 2)  # 확산성과 증식률
        )

        # 부피 예측을 위한 회귀 헤드
        self.volume_head = nn.Sequential(
            nn.Linear(130, 64),  # 128 (특징) + 2 (매개변수)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 1)  # 미래 부피
        )

    def forward(self, x):
        # 특징 추출
        features = self.encoder(x)

        # 세포 밀도 재구성
        density_map = self.density_decoder(features)

        # 전역 특징 추출
        global_features = self.global_pool(features).view(features.size(0), -1)

        # 매개변수 추정
        parameters = self.parameter_head(global_features)

        # 매개변수와 전역 특징 결합
        combined_features = torch.cat([global_features, parameters], dim=1)

        # 부피 예측
        volume_pred = self.volume_head(combined_features)

        return {
            'density_map': density_map,
            'parameters': parameters,
            'volume': volume_pred
        }

class GliomaGrowthPredictor:
    """
    뇌종양 성장 예측 시스템 메인 클래스
    """
    def __init__(self, model_config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ReactionDiffusionNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

    def multi_task_loss(self, predictions, targets, alpha=1.0, beta=0.5, gamma=0.5):
        """
        다중 작업 손실 함수
        - 세포 밀도 재구성 손실
        - 매개변수 추정 손실
        - 부피 예측 손실
        """
        density_loss = F.mse_loss(predictions['density_map'], targets['density_target'])
        param_loss = F.mse_loss(predictions['parameters'], targets['parameters'])
        volume_loss = F.mse_loss(predictions['volume'], targets['future_volume'])

        total_loss = alpha * density_loss + beta * param_loss + gamma * volume_loss

        return total_loss, {
            'density_loss': density_loss.item(),
            'param_loss': param_loss.item(),
            'volume_loss': volume_loss.item(),
            'total_loss': total_loss.item()
        }

    def train_epoch(self, train_loader):
        """
        한 에포크 훈련
        """
        self.model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            # 데이터를 GPU로 이동
            mri = batch['mri'].to(self.device)
            contour_1 = batch['contour_1'].to(self.device)
            contour_2 = batch['contour_2'].to(self.device)

            # 입력 데이터 결합 (MRI + 윤곽선)
            input_data = torch.cat([mri, contour_1.unsqueeze(1), contour_2.unsqueeze(1)], dim=1)

            # 타겟 데이터 준비
            targets = {
                'density_target': batch['contour_2'].to(self.device).unsqueeze(1),  # 더 큰 윤곽선을 타겟으로
                'parameters': torch.stack([batch['diffusivity'], batch['proliferation']], dim=1).to(self.device),
                'future_volume': batch['future_volume'].to(self.device)
            }

            # 순전파
            self.optimizer.zero_grad()
            predictions = self.model(input_data)

            # 손실 계산
            loss, loss_dict = self.multi_task_loss(predictions, targets)

            # 역전파
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_losses.append(loss_dict)

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss = {loss.item():.4f}')

        return epoch_losses

    def validate(self, val_loader):
        """
        검증
        """
        self.model.eval()
        val_losses = []
        volume_predictions = []
        volume_targets = []

        with torch.no_grad():
            for batch in val_loader:
                mri = batch['mri'].to(self.device)
                contour_1 = batch['contour_1'].to(self.device)
                contour_2 = batch['contour_2'].to(self.device)

                input_data = torch.cat([mri, contour_1.unsqueeze(1), contour_2.unsqueeze(1)], dim=1)

                targets = {
                    'density_target': batch['contour_2'].to(self.device).unsqueeze(1),
                    'parameters': torch.stack([batch['diffusivity'], batch['proliferation']], dim=1).to(self.device),
                    'future_volume': batch['future_volume'].to(self.device)
                }

                predictions = self.model(input_data)
                loss, loss_dict = self.multi_task_loss(predictions, targets)

                val_losses.append(loss_dict)
                volume_predictions.extend(predictions['volume'].cpu().numpy())
                volume_targets.extend(targets['future_volume'].cpu().numpy())

        # 부피 예측 정확도 계산
        volume_mse = mean_squared_error(volume_targets, volume_predictions)
        volume_mae = np.mean(np.abs(np.array(volume_targets) - np.array(volume_predictions)))

        return val_losses, volume_mse, volume_mae

    def predict_future_volume(self, mri_path, contour_threshold_1=0.8, contour_threshold_2=0.3):
        """
        단일 MRI에서 미래 종양 부피 예측
        """
        self.model.eval()

        # MRI 데이터 로딩
        if mri_path.endswith('.npy'):
            mri_data = np.load(mri_path)
        else:
            # NIfTI 파일 처리
            nii_data = nib.load(mri_path)
            mri_data = nii_data.get_fdata()

        # 윤곽선 추출
        segmentation = mri_data[-1]
        contour_1 = (segmentation > contour_threshold_1).astype(np.float32)
        contour_2 = (segmentation > contour_threshold_2).astype(np.float32)

        # 텐서 변환 및 배치 차원 추가
        mri_tensor = torch.FloatTensor(mri_data).unsqueeze(0).to(self.device)
        contour_1_tensor = torch.FloatTensor(contour_1).unsqueeze(0).unsqueeze(0).to(self.device)
        contour_2_tensor = torch.FloatTensor(contour_2).unsqueeze(0).unsqueeze(0).to(self.device)

        input_data = torch.cat([mri_tensor, contour_1_tensor, contour_2_tensor], dim=1)

        with torch.no_grad():
            predictions = self.model(input_data)

            future_volume = predictions['volume'].cpu().numpy()[0, 0]
            estimated_params = predictions['parameters'].cpu().numpy()[0]
            density_map = predictions['density_map'].cpu().numpy()[0, 0]

        return {
            'future_volume': future_volume,
            'diffusivity': estimated_params[0],
            'proliferation': estimated_params[1],
            'density_map': density_map
        }

    def train(self, train_loader, val_loader, num_epochs=100):
        """
        모델 훈련
        """
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')

            # 훈련
            train_losses = self.train_epoch(train_loader)
            avg_train_loss = np.mean([loss['total_loss'] for loss in train_losses])

            # 검증
            val_losses, volume_mse, volume_mae = self.validate(val_loader)
            avg_val_loss = np.mean([loss['total_loss'] for loss in val_losses])

            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}')
            print(f'Volume MSE: {volume_mse:.4f}')
            print(f'Volume MAE: {volume_mae:.4f}')

            # 학습률 조정
            self.scheduler.step(avg_val_loss)

            # 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, 'best_glioma_model.pth')
                print(f'Best model saved with validation loss: {avg_val_loss:.4f}')

def main():
    """
    메인 실행 함수
    """
    # 데이터셋 경로 설정
    data_dir = "path/to/synthetic_tumor_data"
    train_csv = "train_data.csv"
    val_csv = "val_data.csv"

    # 데이터셋 생성
    train_dataset = GliomaVolumeDataset(data_dir, train_csv)
    val_dataset = GliomaVolumeDataset(data_dir, val_csv)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # 모델 초기화 및 훈련
    predictor = GliomaGrowthPredictor()
    predictor.train(train_loader, val_loader, num_epochs=100)

    # 예측 예시
    test_mri_path = "path/to/test_mri.npy"
    result = predictor.predict_future_volume(test_mri_path)

    print(f"\n=== 예측 결과 ===")
    print(f"미래 종양 부피: {result['future_volume']:.2f} mm³")
    print(f"추정 확산성: {result['diffusivity']:.4f}")
    print(f"추정 증식률: {result['proliferation']:.4f}")

if __name__ == "__main__":
    main()