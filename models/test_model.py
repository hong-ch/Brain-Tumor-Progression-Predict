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

class EnhancedBrainTumorDataset(Dataset):
    def __init__(self, root_dir, csv_path, augment=False):
        self.root_dir = root_dir
        self.augment = augment
        self.meta_df = pd.read_csv(csv_path)
        self.meta_df['Patient ID'] = self.meta_df['Patient ID'].astype(str)
        self.days_gap_dict = dict(zip(self.meta_df['Patient ID'], self.meta_df.get('Days Gap', pd.Series(1, index=self.meta_df.index))))
        self.sequences, self.patient_ids = [], []
        self.size_stats, self.growth_stats = {}, {}
        self._build_sequences()
        self._calculate_task_stats()

    def _validate_folder(self, path):
        return all(os.path.exists(os.path.join(path, f)) for f in ['flair.nii.gz', 't1ce.nii.gz', 'mask.nii.gz'])

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

    def _calculate_task_stats(self):
        sizes, growths = [], []
        for seq in self.sequences:
            mask = nib.load(os.path.join(seq['input_path'], 'mask.nii.gz')).get_fdata()[seq['slice_idx']]
            target = nib.load(os.path.join(seq['target_path'], 'mask.nii.gz')).get_fdata()[seq['slice_idx']]
            size0, size1 = np.sum(mask), np.sum(target)
            growth = (size1 - size0) / seq['days_gap']
            sizes.append(size1)
            growths.append(growth)
        self.size_stats = {'min': np.min(sizes), 'max': np.max(sizes)}
        self.growth_stats = {'mean': np.mean(growths), 'std': np.std(growths)}

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
        flair = nib.load(os.path.join(seq['input_path'], 'flair.nii.gz')).get_fdata()[seq['slice_idx']]
        t1ce = nib.load(os.path.join(seq['input_path'], 't1ce.nii.gz')).get_fdata()[seq['slice_idx']]
        mask = self._preprocess_mask(nib.load(os.path.join(seq['input_path'], 'mask.nii.gz')).get_fdata()[seq['slice_idx']])
        target = self._preprocess_mask(nib.load(os.path.join(seq['target_path'], 'mask.nii.gz')).get_fdata()[seq['slice_idx']])

        input_tensor = torch.stack([torch.tensor(flair), torch.tensor(t1ce), torch.tensor(mask)])
        if self.augment:
            input_tensor = self._augment_tensor(input_tensor)

        size0, size1 = np.sum(mask), np.sum(target)
        raw_growth = (size1 - size0) / seq['days_gap']

        denom_size = self.size_stats['max'] - self.size_stats['min']
        denom_growth = self.growth_stats['std']

        norm_size = (size1 - self.size_stats['min']) / denom_size if denom_size != 0 else 0.0
        norm_growth = (raw_growth - self.growth_stats['mean']) / denom_growth if denom_growth != 0 else 0.0

        return input_tensor.float(), torch.tensor([norm_size], dtype=torch.float32), torch.tensor([norm_growth], dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

class HybridTumorPredictor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.size_branch = nn.Sequential(
            *list(resnet50().layer1.children()),
            *list(resnet50().layer2.children()),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.growth_branch = nn.Sequential(
            *list(resnet50().layer1.children()),
            *list(resnet50().layer2.children()),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.size_head = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
        self.growth_head = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.shared_conv(x)
        s_feat = self.size_branch(x)
        g_feat = self.growth_branch(x)
        s_feat = s_feat.squeeze(-1).squeeze(-1)
        g_feat = g_feat.squeeze(-1).squeeze(-1)
        return self.size_head(s_feat), self.growth_head(g_feat)

def dynamic_weighted_loss(p1, p2, y1, y2, epoch, max_epoch):
    mse = nn.MSELoss(reduction='none')
    huber = nn.HuberLoss(reduction='none')
    size_scale = torch.std(y1.detach(), unbiased=False)
    size_scale = size_scale if size_scale > 1e-6 else torch.tensor(1.0, device=y1.device)
    growth_scale = torch.std(y2.detach(), unbiased=False)
    growth_scale = growth_scale if growth_scale > 1e-6 else torch.tensor(1.0, device=y2.device)
    progress = epoch / max_epoch
    rare_weight = 1.0 + (3.0 - 1.0) * (1 - torch.cos(torch.tensor(progress * np.pi))) / 2
    s_rare = ((y1 < torch.quantile(y1, 0.1)) | (y1 > torch.quantile(y1, 0.9))).float()
    g_rare = ((y2 < torch.quantile(y2, 0.1)) | (y2 > torch.quantile(y2, 0.9))).float()
    l_size = (rare_weight * s_rare * mse(p1, y1) / size_scale).mean()
    l_growth = (rare_weight * g_rare * huber(p2, y2) / growth_scale).mean()
    return l_size + 1.2 * l_growth
    
# --- EarlyStopping ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# --- Train Function ---
def train_hybrid_model(data_dir, csv_path, epochs=30, batch_size=8, device=None, patience=5):
    dataset = EnhancedBrainTumorDataset(data_dir, csv_path, augment=True)
    patients = sorted(list(set(dataset.patient_ids)))
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
    train_val_idx, test_idx = next(gss1.split(patients, groups=patients))
    train_val_pids = [patients[i] for i in train_val_idx]
    test_pids = [patients[i] for i in test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
    train_idx, val_idx = next(gss2.split(train_val_pids, groups=train_val_pids))
    train_pids = [train_val_pids[i] for i in train_idx]
    val_pids = [train_val_pids[i] for i in val_idx]

    train_indices = [i for i, pid in enumerate(dataset.patient_ids) if pid in train_pids]
    val_indices = [i for i, pid in enumerate(dataset.patient_ids) if pid in val_pids]
    test_indices = [i for i, pid in enumerate(dataset.patient_ids) if pid in test_pids]

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridTumorPredictor().to(device)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for batch_idx, (x, y1, y2) in enumerate(train_loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            opt.zero_grad()
            p1, p2 = model(x)
            loss = dynamic_weighted_loss(p1, p2, y1, y2, epoch, epochs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_train_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Train Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()

        # Validation step (배치별 loss 출력)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, (x_val, y1_val, y2_val) in enumerate(val_loader):
                x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                p1_val, p2_val = model(x_val)
                batch_loss = dynamic_weighted_loss(p1_val, p2_val, y1_val, y2_val, epoch, epochs)
                val_loss += batch_loss.item()
                print(f"Epoch {epoch+1}/{epochs} | Val Batch {val_batch_idx+1}/{len(val_loader)} | Loss: {batch_loss.item():.4f}")
        avg_val_loss = val_loss / len(val_loader)

        # Early stopping 체크
        early_stopping(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Avg Val Loss: {avg_val_loss:.4f} | EarlyStop Counter: {early_stopping.counter}")
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Test set 평가
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for test_batch_idx, (x_test, y1_test, y2_test) in enumerate(test_loader):
            x_test, y1_test, y2_test = x_test.to(device), y1_test.to(device), y2_test.to(device)
            p1_test, p2_test = model(x_test)
            batch_loss = dynamic_weighted_loss(p1_test, p2_test, y1_test, y2_test, epoch, epochs)
            test_loss += batch_loss.item()
            print(f"Test Batch {test_batch_idx+1}/{len(test_loader)} | Loss: {batch_loss.item():.4f}")
    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Avg Loss: {avg_test_loss:.4f}")

    return model, (train_loader, val_loader, test_loader), dataset

# --- 평가 및 시각화 ---
def evaluate_model(model, loader, device, dataset):
    model.eval()
    size_pred, size_true = [], []
    growth_pred, growth_true = [], []

    with torch.no_grad():
        for x, y1, y2 in loader:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            p1, p2 = model(x)
            size_pred.extend(p1.cpu().numpy())
            size_true.extend(y1.cpu().numpy())
            growth_pred.extend(p2.cpu().numpy())
            growth_true.extend(y2.cpu().numpy())

    size_true = np.array(size_true) * (dataset.size_stats['max'] - dataset.size_stats['min']) + dataset.size_stats['min']
    size_pred = np.array(size_pred) * (dataset.size_stats['max'] - dataset.size_stats['min']) + dataset.size_stats['min']
    growth_true = np.array(growth_true) * dataset.growth_stats['std'] + dataset.growth_stats['mean']
    growth_pred = np.array(growth_pred) * dataset.growth_stats['std'] + dataset.growth_stats['mean']

    size_r2 = 1 - np.sum((size_true - size_pred)**2) / np.sum((size_true - np.mean(size_true))**2)
    growth_mape = np.mean(np.abs((growth_true - growth_pred) / (growth_true + 1e-8))) * 100

    return {
        'size_r2': size_r2,
        'growth_mape': growth_mape,
        'size_pred': size_pred,
        'size_true': size_true,
        'growth_pred': growth_pred,
        'growth_true': growth_true
    }

def plot_enhanced_results(results):
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    sns.regplot(x=results['size_true'], y=results['size_pred'], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f"Size Prediction (R²={results['size_r2']:.3f})")
    plt.xlabel('Actual Size')
    plt.ylabel('Predicted Size')
    plt.subplot(1,2,2)
    sns.histplot(results['growth_true'] - results['growth_pred'], kde=True, bins=30)
    plt.title(f"Growth Rate Error Distribution (MAPE={results['growth_mape']:.2f}%)")
    plt.xlabel('Prediction Error')
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == '__main__':
    data_dir = "/mnt/ssd/brain-tumor-prediction/data/btp_reg_flirt_all_fixed22_nostrip"
    csv_path = "/mnt/ssd/brain-tumor-prediction/csv/btp_meta.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model, loaders, dataset = train_hybrid_model(data_dir, csv_path, device=device, patience=5)
    eval_results = evaluate_model(trained_model, loaders[2], device, dataset)
    plot_enhanced_results(eval_results)
