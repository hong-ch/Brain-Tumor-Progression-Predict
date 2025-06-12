import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

# --- 모델 하이퍼파라미터 ---
N_SLICES = 22
IMG_HEIGHT = 240
IMG_WIDTH = 240

# --- 모델 정의 ---

def build_efficientnet_pooling_model():
    """EfficientNet과 Global Pooling을 사용한 모델을 구성합니다."""
    
    # 1. 입력층 정의
    # Input shape: (num_slices, height, width, 1)
    input_layer = Input(shape=(N_SLICES, IMG_HEIGHT, IMG_WIDTH, 1), name="mri_input")

    # 2. 채널 조정 (1-channel MRI -> 3-channel for EfficientNet)
    # 각 슬라이스에 대해 Conv2D를 적용하기 위해 TimeDistributed 사용
    x = TimeDistributed(Conv2D(3, (1, 1), padding='same', activation='relu'), name="channel_adapter")(input_layer)

    # 3. EfficientNet 백본 (사전 훈련된 가중치 사용, 상단 분류층 제외)
    backbone = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    backbone.trainable = True # 필요에 따라 fine-tuning 여부 결정
    
    # 각 슬라이스에 백본 적용
    x = TimeDistributed(backbone, name="efficientnet_backbone")(x)
    
    # EfficientNet의 출력을 벡터화
    x = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(), name="slice_feature_pooling")(x)

    # 4. 전역 풀링 (슬라이스들의 특징을 평균)
    # (None, 22, 1280) -> (None, 1280)
    aggregated_features = GlobalAveragePooling1D(name="aggregate_slice_features")(x)

    # 5. 회귀 헤드 (MLP)
    x = Dropout(0.3)(aggregated_features)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1, activation='linear', name="volume_output")(x) # 최종 부피 예측

    # 모델 생성
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# 모델 생성 및 요약 출력
model_a = build_efficientnet_pooling_model()
model_a.summary()

# 모델 컴파일
model_a.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='mean_squared_error', # 회귀 문제이므로 MSE 사용
                metrics=['mean_absolute_error'])