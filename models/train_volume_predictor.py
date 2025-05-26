#!/usr/bin/env python3
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from scipy import ndimage


class GliomaVolumeDataset(Dataset):
    def __init__(self, preproc_root, reg_root, meta_csv, mods=None):
        """
        preproc_root: 전처리된 원본(고정) 데이터 루ート
        reg_root:     FLIRT 정합 결과 루트
        meta_csv:     btp_meta.csv 경로
        mods:         ['flair','t1','t1ce','t2'] 등 사용 모달리티
        """
        self.preproc_root = preproc_root
        self.reg_root     = reg_root
        self.mods         = mods or ["flair","t1","t1ce","t2"]
        self.mask_name    = "mask.nii.gz"
        self.reg_mask_name= "mask.nii.gz"

        # 1) 메타 CSV 로딩
        self.meta = pd.read_csv(meta_csv)

        # 2) 컬럼명 통일: strip → lower → 공백/하이픈/점 → underscore
        self.meta.columns = (
            self.meta.columns
            .str.strip()
            .str.lower()
            .str.replace(r"[ \-\.]+", "_", regex=True)
        )
        # 이제 컬럼명은 e.g. patient_id, date_1, date_2, days_gap, age_1, age_2,
        # diffusivity, proliferation 등으로 바뀜

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        pid       = row["patient_id"]
        fixed_dt  = row["date_1"]    # YYYY-MM-DD
        moving_dt = row["date_2"]    # YYYY-MM-DD
        days_gap  = row["days_gap"]
        age1      = row["age_1"]
        age2      = row["age_2"]

        # 1) 고정 세션 MRI 모달리티 읽기 (Z,Y,X)
        fixed_vols = []
        for m in self.mods:
            p = os.path.join(self.preproc_root, pid, fixed_dt, f"{m}.nii.gz")
            fixed_vols.append(nib.load(p).get_fdata())
        fixed_vols = np.stack(fixed_vols, axis=0).astype(np.float32)
        # shape: (C, Z, Y, X)

        # 2) 고정 세션 마스크 읽기
        fixed_mask = nib.load(
            os.path.join(self.preproc_root, pid, fixed_dt, self.mask_name)
        ).get_fdata().astype(np.uint8)

        # 3) 정합된 이동 세션 마스크 읽기
        reg_mask = nib.load(
            os.path.join(self.reg_root, pid, moving_dt, self.reg_mask_name)
        ).get_fdata().astype(np.uint8)

        # 4) 미래 부피 계산 (voxel volume × mask voxel 수)
        hdr = nib.load(
            os.path.join(self.reg_root, pid, moving_dt, self.reg_mask_name)
        ).header
        vx, vy, vz = hdr.get_zooms()[:3]
        voxel_vol  = vx * vy * vz
        future_volume = float(reg_mask.sum()) * voxel_vol

        sample = {
            "fixed_vols":      torch.from_numpy(fixed_vols),
            "fixed_mask":      torch.from_numpy(fixed_mask)[None].float(),
            "moving_reg_mask": torch.from_numpy(reg_mask)[None].float(),
            "future_volume":   torch.tensor([future_volume], dtype=torch.float32),
            "days_gap":        torch.tensor([days_gap], dtype=torch.float32),
            "age1":            torch.tensor([age1], dtype=torch.float32),
            "age2":            torch.tensor([age2], dtype=torch.float32),
            "diffusivity":     torch.tensor([row["diffusivity"]], dtype=torch.float32),
            "proliferation":   torch.tensor([row["proliferation"]], dtype=torch.float32),
        }
        return sample


class ReactionDiffusionNet(nn.Module):
    """
    반응-확산 뇌종양 성장 모델용 3D CNN
    """
    def __init__(self, input_channels=4):
        super().__init__()
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        # --- Decoder (Density map) ---
        self.density_decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # --- Parameter head ---
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.parameter_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 2)  # diffusivity, proliferation
        )
        # --- Volume head: features(128)+params(2)+meta(2) ---
        self.volume_head = nn.Sequential(
            nn.Linear(128 + 2 + 2, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x, age, days_gap):
        feats = self.encoder(x)
        density = self.density_decoder(feats)
        gf = self.global_pool(feats).view(feats.size(0), -1)  # (B,128)
        params = self.parameter_head(gf)                     # (B,2)
        meta   = torch.cat([age, days_gap], dim=1)           # (B,2)
        cat_in = torch.cat([gf, params, meta], dim=1)        # (B,132)
        volume = self.volume_head(cat_in)                    # (B,1)
        return {"density_map": density, "parameters": params, "volume": volume}


class GliomaGrowthPredictor:
    """
    뇌종양 성장 예측 훈련/검증 루틴
    """
    def __init__(self, lr):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = ReactionDiffusionNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

    def multi_task_loss(self, pred, targ, α=1.0, β=0.5, γ=0.5):
        d_loss = F.mse_loss(pred["density_map"], targ["density_target"])
        p_loss = F.mse_loss(pred["parameters"],   targ["parameters"])
        v_loss = F.mse_loss(pred["volume"],       targ["future_volume"])
        total  = α*d_loss + β*p_loss + γ*v_loss
        return total, {"density":d_loss.item(), "param":p_loss.item(), "volume":v_loss.item(), "total":total.item()}

    def train_one_epoch(self, loader):
        self.model.train()
        losses = []
        for batch in loader:
            self.optimizer.zero_grad()
            x   = torch.cat([batch["fixed_vols"],
                             batch["fixed_mask"],    # as channel
                             batch["moving_reg_mask"]], dim=1).to(self.device)
            # density_target 은 moving_reg_mask 와 같도록
            targ = {
                "density_target": batch["moving_reg_mask"].to(self.device),
                "parameters":     torch.stack([batch["diffusivity"], batch["proliferation"]], dim=1).to(self.device),
                "future_volume":  batch["future_volume"].to(self.device)
            }
            age      = batch["age1"].to(self.device) .unsqueeze(1)
            days_gap = batch["days_gap"].to(self.device).unsqueeze(1)
            pred     = self.model(x, age, days_gap)
            loss, _  = self.multi_task_loss(pred, targ)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def validate(self, loader):
        self.model.eval()
        losses, vols_p, vols_t = [], [], []
        with torch.no_grad():
            for batch in loader:
                x   = torch.cat([batch["fixed_vols"],
                                 batch["fixed_mask"],
                                 batch["moving_reg_mask"]], dim=1).to(self.device)
                targ = {
                    "density_target": batch["moving_reg_mask"].to(self.device),
                    "parameters":     torch.stack([batch["diffusivity"], batch["proliferation"]], dim=1).to(self.device),
                    "future_volume":  batch["future_volume"].to(self.device)
                }
                age      = batch["age1"].to(self.device).unsqueeze(1)
                days_gap = batch["days_gap"].to(self.device).unsqueeze(1)
                pred     = self.model(x, age, days_gap)
                loss, _  = self.multi_task_loss(pred, targ)
                losses.append(loss.item())
                vols_p.extend(pred["volume"].cpu().numpy().ravel())
                vols_t.extend(targ["future_volume"].cpu().numpy().ravel())
        mse = mean_squared_error(vols_t, vols_p)
        return float(np.mean(losses)), mse

    def fit(self, tr_loader, va_loader, epochs, out_path):
        best_mse = float("inf")
        for ep in range(1, epochs+1):
            tr_loss = self.train_one_epoch(tr_loader)
            va_loss, va_mse = self.validate(va_loader)
            self.scheduler.step(va_mse)
            print(f"[Epoch {ep:02d}] Train Loss: {tr_loss:.4f} | Val MSE: {va_mse:.4f}")
            if va_mse < best_mse:
                best_mse = va_mse
                torch.save(self.model.state_dict(), out_path)
        print(f"✅ Training complete. Best Val MSE: {best_mse:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_root", required=True)
    p.add_argument("--reg_root",     required=True)
    p.add_argument("--meta_csv",     required=True)
    p.add_argument("--batch",  type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--model_out", required=True)
    args = p.parse_args()

    # Dataset & Split
    ds = GliomaVolumeDataset(
        preproc_root = args.preproc_root,
        reg_root     = args.reg_root,
        meta_csv     = args.meta_csv,
        mods         = ["flair","t1","t1ce","t2"]
    )
    from sklearn.model_selection import train_test_split
    idx_tr, idx_va = train_test_split(list(range(len(ds))),
                                      test_size=args.test_size,
                                      random_state=42)
    tr_loader = DataLoader(torch.utils.data.Subset(ds, idx_tr),
                           batch_size=args.batch, shuffle=True,  num_workers=4)
    va_loader = DataLoader(torch.utils.data.Subset(ds, idx_va),
                           batch_size=args.batch, shuffle=False, num_workers=4)

    # Train
    pred = GliomaGrowthPredictor(lr=args.lr)
    pred.fit(tr_loader, va_loader, epochs=args.epochs, out_path=args.model_out)


if __name__=="__main__":
    main()
