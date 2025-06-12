import torch
import torch.nn as nn
import timm

# --- 데이터 스펙에 맞춘 모델 하이퍼파라미터 ---
N_SLICES = 22
IMG_HEIGHT = 240
IMG_WIDTH = 240

class EfficientNetRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # 1채널(MRI)을 3채널(EfficientNet 입력)로 변환
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, padding=0)
        
        # timm 라이브러리에서 사전 훈련된 EfficientNet-B0 로드
        # num_classes=0으로 설정하면 분류기(head)를 제외하고 특징 추출 부분만 가져옴
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained, 
            in_chans=3, 
            num_classes=0, # This returns features before the classifier
            global_pool='' # We will do our own pooling
        )
        # EfficientNet-B0의 특징 벡터 크기는 1280
        feature_dim = self.backbone.num_features
        
        # 회귀 헤드 (MLP)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_slices, C, H, W) -> (B, 22, 1, 240, 240)
        batch_size, num_slices, C, H, W = x.shape
        
        # TimeDistributed 구현: 모든 슬라이스를 하나의 배치처럼 처리
        x = x.view(batch_size * num_slices, C, H, W)
        
        # 채널 조정 및 특징 추출
        x = self.channel_adapter(x)
        features = self.backbone(x) # (B*S, feature_dim, H', W')

        # Global Average Pooling 2D
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1) # (B*S, feature_dim)
        
        # 원래의 배치와 슬라이스 차원으로 복원
        features = features.view(batch_size, num_slices, -1) # (B, S, feature_dim)
        
        # 슬라이스들의 특징을 평균 (Global Pooling 1D)
        aggregated_features = torch.mean(features, dim=1) # (B, feature_dim)
        
        # 최종 부피 예측
        output = self.regression_head(aggregated_features)
        
        return output.squeeze(1) # (B, 1) -> (B)