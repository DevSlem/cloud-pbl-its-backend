# app.py
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from pipeline import TrafficPredictionPipeline

app = FastAPI()

# Load the model and pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = TrafficPredictionPipeline(
    model_path="traffic_lstm_with_embeddings.pt",
    scaler_path="scaler.pkl",
    device=device
)
pipeline.load_model()

# Define the valid date range
TRAIN_START_DATE = datetime(2021, 1, 1)
TRAIN_END_DATE = datetime(2021, 3, 31)
# Define the input data model with validation
class TrafficInput(BaseModel):
    날짜: str  # Date in YYYY-MM-DD format
    시간: int  # Hour in 0-23
    도로명: str
    지점명: str

    @validator('날짜')
    def validate_date(cls, v):
        try:
            date_obj = datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError('날짜 must be in YYYY-MM-DD format')

        # Check if date is within the valid range
        if not (TRAIN_START_DATE <= date_obj <= TRAIN_END_DATE):
            raise ValueError(f"날짜 must be between {TRAIN_START_DATE.strftime('%Y-%m-%d')} and {TRAIN_END_DATE.strftime('%Y-%m-%d')}")

        return v

    @validator('시간')
    def validate_time(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('시간 must be between 0 and 23')
        return v


# Define the output data model
class TrafficOutput(BaseModel):
    predictions: dict

# API endpoint
@app.post("/predict", response_model=TrafficOutput)
def predict_traffic(input_data: TrafficInput):
    # Extract input data
    date_str = input_data.날짜
    time = input_data.시간
    road_name = input_data.도로명
    spot_name = input_data.지점명

    # Validate date
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Validate road_name and spot_name
    if road_name not in pipeline.road_vocab:
        raise HTTPException(status_code=400, detail=f"도로명 '{road_name}' is not recognized.")
    if spot_name not in pipeline.spot_vocab:
        raise HTTPException(status_code=400, detail=f"지점명 '{spot_name}' is not recognized.")

    # Perform prediction
    try:
        result = pipeline.predict(
            year=date_obj.year,
            month=date_obj.month,
            day=date_obj.day,
            road_name=road_name,
            spot_name=spot_name,
            current_hour=time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Prepare output
    formatted_result = {k: float(v) for k, v in result.items()}

    return TrafficOutput(predictions=formatted_result)

@app.get("/roads")
def get_road_names():
    road_names = list(pipeline.road_vocab.keys())
    return JSONResponse(content={"road_names": road_names})

@app.get("/locations")
def get_location_names():
    location_names = list(pipeline.spot_vocab.keys())
    return JSONResponse(content={"location_names": location_names})