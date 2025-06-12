import os
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet50
from torchvision.transforms import functional as TF
from sklearn.model_selection import GroupShuffleSplit
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
class MultiTaskBrainTumor2DSliceDataset(Dataset):
    def __init__(self, root_dir, csv_path, augment=False, resize_size=224):
        self.root_dir = root_dir
        self.augment = augment
        self.meta_df = pd.read_csv(csv_path)
        self.meta_df['Patient ID'] = self.meta_df['Patient ID'].astype(str)
        self.days_gap_dict = dict(zip(self.meta_df['Patient ID'], self.meta_df.get('Days Gap', pd.Series(1, index=self.meta_df.index))))
        self.sequences, self.patient_ids = [], []
        self._build_sequences()

    def _validate_folder(self, path):
        return all(os.path.exists(os.path.join(path, f)) for f in ['flair.nii.gz', 't1ce.nii.gz', 't2.nii.gz', 't1.nii.gz', 'mask.nii.gz'])

    def _get_valid_timepoints(self, patient_path):
        return [{'date': d, 'path': os.path.join(patient_path, d)} for d in sorted(os.listdir(patient_path))
                if os.path.isdir(os.path.join(patient_path, d)) and self._validate_folder(os.path.join(patient_path, d))]

    def _build_sequences(self):
        for _, row in self.meta_df.iterrows():
            pid = str(row['Patient ID'])
            ppath = os.path.join(self.root_dir, pid)
            times = self._get_valid_timepoints(ppath)
            if len(times) < 2:
                continue
            gap = self.days_gap_dict.get(pid, 1)
            for i in range(len(times) - 1):
                im1 = nib.load(os.path.join(times[i]['path'], 'mask.nii.gz')).get_fdata()
                im2 = nib.load(os.path.join(times[i + 1]['path'], 'mask.nii.gz')).get_fdata()
                for z in range(22):
                    if np.sum(im1[z]) == 0 and np.sum(im2[z]) == 0:
                        continue
                    self.sequences.append({'input_path': times[i]['path'], 'target_path': times[i + 1]['path'],
                                           'patient_id': pid, 'slice_idx': z, 'days_gap': gap})
                    self.patient_ids.append(pid)

    def _preprocess_mask(self, m):
        return (np.nan_to_num(m, nan=0.0) > 0).astype(np.float32)

    def _augment_tensor(self, t):
        if random.random() < 0.5:
            t = TF.hflip(t)
        if random.random() < 0.5:
            t = TF.vflip(t)
        angle = random.uniform(-15, 15)
        t = TF.rotate(t, angle)
        return t

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        flair = torch.tensor(nib.load(os.path.join(seq['input_path'], 'flair.nii.gz')).get_fdata()[seq['slice_idx']])
        t1ce = torch.tensor(nib.load(os.path.join(seq['input_path'], 't1ce.nii.gz')).get_fdata()[seq['slice_idx']])
        t2 = torch.tensor(nib.load(os.path.join(seq['input_path'], 't2.nii.gz')).get_fdata()[seq['slice_idx']])
        t1 = torch.tensor(nib.load(os.path.join(seq['input_path'], 't1.nii.gz')).get_fdata()[seq['slice_idx']])
        mask = torch.tensor(self._preprocess_mask(nib.load(os.path.join(seq['input_path'], 'mask.nii.gz')).get_fdata()[seq['slice_idx']]))
        target = torch.tensor(self._preprocess_mask(nib.load(os.path.join(seq['target_path'], 'mask.nii.gz')).get_fdata()[seq['slice_idx']]))

        # resize 관련 코드 제거됨
        input_tensor = torch.stack([flair, t1ce, t2, t1, mask])
        if self.augment:
            input_tensor = self._augment_tensor(input_tensor)

        size0, size1 = mask.sum().item(), target.sum().item()
        growth = (size1 - size0) / seq['days_gap']
        return input_tensor.float(), torch.tensor([size1], dtype=torch.float32), torch.tensor([growth], dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

# Model
class MultiTaskTumorPredictor2D(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.base = resnet50(pretrained=True)
        # 기존 3채널 가중치 추출
        original_weights = self.base.conv1.weight.data.clone()  # (64, 3, 7, 7)
        # 3채널 평균을 내서 5채널로 확장
        new_weights = original_weights.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels
        # 5채널 conv1로 교체 및 가중치 할당
        self.base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base.conv1.weight.data = new_weights

        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.size_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_ftrs, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
        self.growth_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_ftrs, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.base(x)
        return self.size_head(feat), self.growth_head(feat)

# Validation loss 출력 함수 (배치별 번호 포함)
def evaluate_multitask_model(model, loader, device, s_low, s_up, g_low, g_up, rare_weight_factor=2.0, epoch=0):
    model.eval()
    mse, huber = nn.MSELoss(reduction='none'), nn.HuberLoss(reduction='none')
    with torch.no_grad():
        for batch_idx, (x, y1, y2) in enumerate(loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            p1, p2 = model(x)
            s_rare = ((y1.squeeze() < s_low) | (y1.squeeze() > s_up)).float()
            g_rare = ((y2.squeeze() < g_low) | (y2.squeeze() > g_up)).float()
            s_w = 1.0 + s_rare * (rare_weight_factor - 1.0)
            g_w = 1.0 + g_rare * (rare_weight_factor - 1.0)
            l1 = (s_w * mse(p1.squeeze(), y1.squeeze())).mean()
            l2 = (g_w * huber(p2.squeeze(), y2.squeeze())).mean()
            loss = l1 + 0.6 * l2
            print(f"Validation - Epoch: {epoch+1}, Batch: {batch_idx+1}, Size Loss (MSE): {l1.item():.4f}, Growth Loss (Huber): {l2.item():.4f}, Total Loss: {loss.item():.4f}")

# Train
def train_multitask_model(data_dir, csv_path, epochs=30, batch_size=4, lr=3e-4, rare_weight_max=3.0):
    dataset = MultiTaskBrainTumor2DSliceDataset(data_dir, csv_path, augment=True)
    patients = sorted(list(set(dataset.patient_ids)))
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
    train_val_idx, test_idx = next(gss1.split(patients, groups=patients))
    train_val_pids = [patients[i] for i in train_val_idx]
    test_pids = [patients[i] for i in test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
    train_idx, val_idx = next(gss2.split(train_val_pids, groups=train_val_pids))
    train_pids = [train_val_pids[i] for i in train_idx]
    val_pids = [train_val_pids[i] for i in val_idx]

    print(f"\n[환자별 데이터 분할 정보]")
    print(f"Train 환자 수: {len(train_pids)}")
    print(f"Validation 환자 수: {len(val_pids)}")
    print(f"Test 환자 수: {len(test_pids)}\n")

    train_indices = [i for i, pid in enumerate(dataset.patient_ids) if pid in train_pids]
    val_indices = [i for i, pid in enumerate(dataset.patient_ids) if pid in val_pids]
    test_indices = [i for i, pid in enumerate(dataset.patient_ids) if pid in test_pids]

    sizes, growths = [], []
    for i in train_indices:
        _, s, g = dataset[i]
        sizes.append(s.item())
        growths.append(g.item())

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size)

    model = MultiTaskTumorPredictor2D().cuda()
    opt = optim.Adam(model.parameters(), lr=lr)
    mse, huber = nn.MSELoss(reduction='none'), nn.HuberLoss(reduction='none')

    for epoch in range(epochs):
        model.train()
        progress = epoch / epochs

        # 동적 가중치 (Cosine schedule)
        rare_weight_factor = 1.0 + (rare_weight_max - 1.0) * (1 - np.cos(progress * np.pi)) / 2

        # rare 범위도 점점 좁히기
        s_low = np.percentile(sizes, 10 + 5 * (1 - progress))
        s_up = np.percentile(sizes, 90 - 5 * (1 - progress))
        g_low = np.percentile(growths, 10 + 5 * (1 - progress))
        g_up = np.percentile(growths, 90 - 5 * (1 - progress))

        for batch_idx, (x, y1, y2) in enumerate(train_loader):
            x, y1, y2 = x.cuda(), y1.cuda(), y2.cuda()
            opt.zero_grad()
            p1, p2 = model(x)

            s_rare = ((y1.squeeze() < s_low) | (y1.squeeze() > s_up)).float()
            g_rare = ((y2.squeeze() < g_low) | (y2.squeeze() > g_up)).float()
            s_w = 1.0 + s_rare * (rare_weight_factor - 1.0)
            g_w = 1.0 + g_rare * (rare_weight_factor - 1.0)

            l1 = (s_w * mse(p1.squeeze(), y1.squeeze())).mean()
            l2 = (g_w * huber(p2.squeeze(), y2.squeeze())).mean()
            loss = l1 + 0.6 * l2
            loss.backward()
            opt.step()

            print(f"Train - Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}, "
                  f"Size Loss: {l1.item():.4f}, Growth Loss: {l2.item():.4f}, Total: {loss.item():.4f}")

        print(f"\n--- Validation for Epoch {epoch+1} ---")
        evaluate_multitask_model(model, val_loader, torch.device("cuda"),
                                 s_low, s_up, g_low, g_up, rare_weight_factor, epoch)
        print("--- End Validation ---\n")

    return model, DataLoader(Subset(dataset, test_indices), batch_size=1)

# Test
def test_multitask_model(model, loader, device):
    model.eval()
    mse, huber = nn.MSELoss(), nn.HuberLoss()
    sp, gp, st, gt = [], [], [], []
    with torch.no_grad():
        for sample_idx, (x, y1, y2) in enumerate(loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            p1, p2 = model(x)
            l1 = mse(p1.squeeze(), y1.squeeze())
            l2 = huber(p2.squeeze(), y2.squeeze())
            loss = l1 + 0.6 * l2
            print(f"Test - Sample: {sample_idx+1}, Size Loss (MSE): {l1.item():.4f}, Growth Loss (Huber): {l2.item():.4f}, Total Loss: {loss.item():.4f}")
            sp.append(p1.item())
            gp.append(p2.item())
            st.append(y1.item())
            gt.append(y2.item())
    return sp, st, gp, gt

# Plot
def plot_multitask_results(sp, st, gp, gt):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=st, y=sp, alpha=0.6)
    plt.plot([min(st), max(st)], [min(st), max(st)], 'r--')
    plt.title("Size Prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=gt, y=gp, alpha=0.6)
    plt.plot([min(gt), max(gt)], [min(gt), max(gt)], 'r--')
    plt.title("Growth Prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main
if __name__ == '__main__':
    data_dir = "/content/drive/MyDrive/btp_reg_flirt_all_224_nostrip"
    csv_path = "/content/drive/MyDrive/btp_meta.csv"
    model, test_loader = train_multitask_model(data_dir, csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sp, st, gp, gt = test_multitask_model(model, test_loader, device)
    plot_multitask_results(sp, st, gp, gt)
