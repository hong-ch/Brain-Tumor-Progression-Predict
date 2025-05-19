#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main():
    # 1) CSV 불러오기
    df = pd.read_csv(os.path.join('../csv', 'btp_meta_volume.csv'))

    # 2) Sex 인코딩: 'M'→0, 'F'→1
    df['sex'] = df['Sex'].map({'M': 0, 'F': 1})

    # 3) Feature / Target 지정
    #    예측 시점에 이미 알 수 있는 정보만 사용
    feature_cols = [
        'Days Gap',
        'Volume 1',
        'sex',
        'Age 1',
        'Age 2',
        'Weight 1',
        'Weight 2'
    ]
    target_col = 'Volume 2'

    X = df[feature_cols]
    y = df[target_col]

    # 4) Train / Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # 5) 모델 초기화 및 학습
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # 6) 성능 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")

    # 7) 학습된 모델 저장
    os.makedirs(os.path.join('models', 'reg'), exist_ok=True)
    save_path = os.path.join('models', 'reg', 'ridge_btp.pkl')
    joblib.dump(model, save_path)
    print(f"Saved trained model to {save_path}")

if __name__ == '__main__':
    main()
