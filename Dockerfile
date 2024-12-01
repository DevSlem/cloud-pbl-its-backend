# 베이스 이미지로 Python 3.9 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt ./
COPY app.py pipeline.py scaler.pkl traffic_lstm_with_embeddings.pt ./

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# uvicorn을 통해 앱 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
