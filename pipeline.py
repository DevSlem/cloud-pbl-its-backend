import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# 1. Scaler 클래스 정의
class Scaler:
    def __init__(self, scaler_path="scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler = None

    def fit(self, data):
        self.scaler = MinMaxScaler()
        self.scaler.fit(data)
        self.save()

    def transform(self, data):
        if self.scaler is None:
            self.load()
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        if self.scaler is None:
            self.load()
        return self.scaler.inverse_transform(data)

    def save(self):
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self):
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scaler file '{self.scaler_path}' not found. Please fit the scaler first.")

class TrafficLSTM(nn.Module):
    def __init__(self, road_vocab_size, spot_vocab_size, hidden_size, num_layers=2, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        # Embedding layers for categorical data
        self.year_embed = nn.Embedding(3000, 2)  # 년 임베딩 (최대 3000년)
        self.month_embed = nn.Embedding(12, 2)   # 월 임베딩 (1~12월)
        self.day_embed = nn.Embedding(31, 2)     # 일 임베딩 (1~31일)
        self.road_embed = nn.Embedding(road_vocab_size, 10)  # 도로명 임베딩
        self.spot_embed = nn.Embedding(spot_vocab_size, 10)  # 지점명 임베딩
        
        # LSTM layers
        self.lstm = nn.LSTM(2 + 2 + 2 + 10 + 10 + 1, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Extract embeddings
        year_ids = x[:, 0].long()  # 년 ID
        month_ids = x[:, 1].long()  # 월 ID
        day_ids = x[:, 2].long()  # 일 ID
        road_ids = x[:, 3].long()  # 도로명 ID
        spot_ids = x[:, 4].long()  # 지점명 ID
        year_embed = self.year_embed(year_ids)
        month_embed = self.month_embed(month_ids)
        day_embed = self.day_embed(day_ids)
        road_embed = self.road_embed(road_ids)
        spot_embed = self.spot_embed(spot_ids)
        
        # Concatenate embeddings with continuous features
        # (batch_size, 24)
        continuous_features = x[:, 5:]  # 시간 데이터 및 기타 연속형 변수
        # (batch_size, 26)
        embed_features = torch.cat([year_embed, month_embed, day_embed, road_embed, spot_embed], dim=1)
        
        # (batch_size, 24, 1)
        continuous_feature_seq = continuous_features.unsqueeze(-1)
        # (batch_size, 24, 26)
        embed_feature_seq = embed_features.unsqueeze(1).expand(-1, 24, -1)
        feature_seq = torch.cat([continuous_feature_seq, embed_feature_seq], dim=-1)
        
        # LSTM forward pass
        out, _ = self.lstm(feature_seq)
        out = self.fc(out)
        return out[:, :, 0]

# 3. 파이프라인 클래스 정의
class TrafficPredictionPipeline:
    def __init__(self, model_path, scaler_path, device):
        self.model_path = model_path
        self.scaler = Scaler(scaler_path)
        self.device = device
        self.model = None

    def load_model(self):
        ckpt = torch.load(self.model_path, map_location=self.device)
        metadata = ckpt['metadata']
        
        self.road_vocab = metadata['road_vocab']
        self.spot_vocab = metadata['spot_vocab']
        
        self.model = TrafficLSTM(
            road_vocab_size=metadata['road_vocab_size'],
            spot_vocab_size=metadata['spot_vocab_size'],
            hidden_size=metadata['hidden_size'],
        )
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()
        
        self.scaler.load()

    def preprocess_input(self, year, month, day, road_name, spot_name):
        data = pd.DataFrame({
            '년': [year], '월': [month], '일': [day],
            '도로명': [road_name], '지점명': [spot_name]
        })
        data['도로명_ID'] = data['도로명'].map(self.road_vocab)
        data['지점명_ID'] = data['지점명'].map(self.spot_vocab)

        time_cols = [f"{i:02d}시" for i in range(24)]
        for col in time_cols:
            data[col] = 0  # 시간 데이터를 비워둠

        features = np.hstack([
            data[['년', '월', '일']].values,
            data[['도로명_ID', '지점명_ID']].values,
            np.zeros((1, 24))  # 시간 데이터는 0으로 초기화
        ])
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def predict(self, year, month, day, road_name, spot_name, current_hour):
        if current_hour < 0 or current_hour >= 24:
            raise ValueError("current_hour should be between 0 and 23.")
        
        input_tensor = self.preprocess_input(year, month, day, road_name, spot_name)

        with torch.no_grad():
            predictions = self.model(input_tensor)
        predictions = predictions.cpu().numpy()
        predictions_rescaled = self.scaler.inverse_transform(predictions)
        today_traffic = predictions_rescaled[0, current_hour:]
        tommorow_traffic = np.zeros((0,))

        if current_hour > 0:
            tommorow = datetime(year, month, day) + timedelta(days=1)
            tommorow_input_tensor = self.preprocess_input(tommorow.year, tommorow.month, tommorow.day, road_name, spot_name)
            
            with torch.no_grad():
                predictions_tommorow = self.model(tommorow_input_tensor)
            predictions_tommorow = predictions_tommorow.cpu().numpy()
            predictions_tommorow_rescaled = self.scaler.inverse_transform(predictions_tommorow)
            tommorow_traffic = predictions_tommorow_rescaled[0, :current_hour]
            
        traffic = np.concatenate([today_traffic, tommorow_traffic])
            
        # 시간대 구성
        base_date = datetime(year, month, day)
        time_labels = [(base_date + timedelta(hours=(current_hour + i))).strftime("%Y-%m-%d %H:%M") for i in range(24)]
        return dict(zip(time_labels, traffic))

# 4. 실행 예제
if __name__ == "__main__":
    pipeline = TrafficPredictionPipeline(
        model_path="traffic_lstm_with_embeddings.pt",
        scaler_path="scaler.pkl",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    pipeline.load_model()

    # 예측 실행
    result = pipeline.predict(2021, 11, 4, "아트센터대로", "커낼워크 D4 오피스텔", 9)
    for time, traffic in result.items():
        print(f"{time}: {traffic:.2f}")