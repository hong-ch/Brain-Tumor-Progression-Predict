import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm

# (★) 실제 서버 경로로만 한 번에 맞춰주세요!
dataset_root = "/mnt/ssd/brain-tumor-prediction/data/btp_reg_flirt_all_fixed22_nostrip"
png_root = "/mnt/ssd/brain-tumor-prediction/data/btp_png_slices"
csv_path = "/mnt/ssd/brain-tumor-prediction/data/slice_cnn_features.csv"

os.makedirs(png_root, exist_ok=True)

feature_rows = []

# 1. NIfTI → PNG
for patient_id in tqdm(os.listdir(dataset_root)):
    patient_path = os.path.join(dataset_root, patient_id)
    if not os.path.isdir(patient_path):
        continue
    for date_folder in os.listdir(patient_path):
        date_path = os.path.join(patient_path, date_folder)
        nii_path = os.path.join(date_path, "mask.nii.gz")
        if not os.path.isfile(nii_path):
            continue

        img = nib.load(nii_path)
        data = img.get_fdata()
        n_slices = data.shape[2]
        for i in range(n_slices):
            slice_img = data[:, :, i]
            # 정규화 (0~255)
            if np.max(slice_img) > np.min(slice_img):
                slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
            slice_img = (slice_img * 255).astype(np.uint8)
            png_fname = f"{patient_id}_{date_folder}_{i:03d}.png"
            png_fpath = os.path.join(png_root, png_fname)
            plt.imsave(png_fpath, slice_img, cmap='gray')

# 2. CNN Feature 추출
cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
img_size = (224, 224)

for fname in tqdm(os.listdir(png_root)):
    if not fname.endswith('.png'):
        continue
    basename = fname.split('.')[0]
    patient_id, date_folder, slice_idx = basename.split('_')
    img_path = os.path.join(png_root, fname)
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = cnn_model.predict(x, verbose=0).flatten()  # 2048차원

    row = {
        "patient_id": patient_id,
        "timepoint": date_folder,
        "slice_idx": int(slice_idx)
    }
    for i, v in enumerate(features):
        row[f"img_feature_{i}"] = v
    feature_rows.append(row)

feature_df = pd.DataFrame(feature_rows)
feature_df.to_csv(csv_path, index=False)
print("완료! 최종 CNN feature CSV:", csv_path)
