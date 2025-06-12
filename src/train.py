import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

## TENSORBOARD 통합 ##
from torch.utils.tensorboard import SummaryWriter

# 작성한 모듈 임포트
from data_loader import BrainTumorDataset
from model import EfficientNetRegressor

# --- 상수 정의 ---
DATA_DIR = '/mnt/ssd/brain-tumor-prediction/data/btp_reg_flirt_t1ce_fixed22_nostrip'
RESULTS_DIR = '../results' 
N_SPLITS = 4
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
PATIENCE = 20

# --- GPU 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 환자 목록 가져오기 ---
all_patient_ids = sorted([p for p in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, p))])

# --- 2. K-Fold 교차 검증 루프 ---
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
overall_best_metrics = {}

for fold_num, (train_indices, val_indices) in enumerate(kfold.split(all_patient_ids), 1):
    print("\n" + "="*50)
    print(f"=============== FOLD {fold_num}/{N_SPLITS} ===============")
    print("="*50)
    
    # --- 3. Fold별 데이터 준비 ---
    train_patient_ids = [all_patient_ids[i] for i in train_indices]
    val_patient_ids = [all_patient_ids[i] for i in val_indices]
    
    train_dataset = BrainTumorDataset(DATA_DIR, train_patient_ids)
    val_dataset = BrainTumorDataset(DATA_DIR, val_patient_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 4. 모델, 손실 함수, 옵티마이저 정의 ---
    model = EfficientNetRegressor().to(device)
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. 콜백 및 TensorBoard Writer 설정 ---
    fold_results_dir = os.path.join(RESULTS_DIR, f'fold_{fold_num}')
    os.makedirs(fold_results_dir, exist_ok=True)
    
    ## TENSORBOARD 통합 ##
    # 각 fold의 로그를 저장할 writer 생성
    writer = SummaryWriter(log_dir=os.path.join(fold_results_dir, 'logs'))

    best_val_loss = float('inf')
    patience_counter = 0

    # --- 6. 훈련 및 검증 루프 ---
    for epoch in range(EPOCHS):
        # 훈련
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix(loss=loss.item())
        epoch_train_loss = running_loss / len(train_dataset)

        # 검증
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                mae = mae_criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_mae += mae.item() * inputs.size(0)
                val_pbar.set_postfix(loss=loss.item(), mae=mae.item())
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_mae = val_mae / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.4f}")

        ## TENSORBOARD 통합 ##
        # 스칼라 값들을 TensorBoard에 기록
        writer.add_scalar('Loss/train', epoch_train_loss, epoch + 1)
        writer.add_scalar('Loss/validation', epoch_val_loss, epoch + 1)
        writer.add_scalar('MAE/validation', epoch_val_mae, epoch + 1)
        
        # 모델 저장
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(fold_results_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1

        # 조기 종료
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}.")
            break
            
    ## TENSORBOARD 통합 ##
    # 하이퍼파라미터와 최종 성능 기록
    hparams = {'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'epochs': epoch + 1}
    final_metrics = {'hparam/best_val_loss': best_val_loss, 'hparam/final_val_mae': epoch_val_mae}
    writer.add_hparams(hparams, final_metrics)
    
    # Writer 닫기
    writer.close()

print("\n모든 Fold에 대한 교차 검증 훈련이 완료되었습니다.")