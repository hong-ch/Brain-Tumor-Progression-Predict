#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main():
    # 1) CSV 로드
    df = pd.read_csv(os.path.join('../csv', 'btp_meta_volume.csv'))
    
    # 2) Sex 인코딩 ('M'→0, 'F'→1)
    df['sex'] = df['Sex'].map({'M': 0, 'F': 1})
    
    # 3) Feature / Target 정의
    feature_cols = [
        'Days Gap',
        'Volume 1',
        'sex',
        'Age 1',
        'Age 2',
        'Weight 1',
        'Weight 2',
    ]
    target_col = 'Volume 2'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 4) Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    
    # 5) 모델 초기화
    # alpha_1/alpha_2: noise precision prior
    # lambda_1/lambda_2: weight precision prior
    model = BayesianRidge(
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        fit_intercept=True,
    )
    
    # 6) 학습
    model.fit(X_train, y_train)
    
    # 7) 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # 8) 모델 저장
    os.makedirs(os.path.join('reg'), exist_ok=True)
    save_path = os.path.join('reg','bayesridge_btp.pkl')
    joblib.dump(model, save_path)
    print(f"Saved Bayesian Ridge model to {save_path}")

if __name__ == '__main__':
    main()
