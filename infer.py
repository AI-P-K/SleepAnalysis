import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import json
import numpy as np
from engine.Architectures import *


# Load run hyperparameters
with open("inference/production_models/hparams.json", "r") as f:
    hparams = json.load(f)

# Load the scaler
scaler = joblib.load("inference/production_models/scaler.pkl")

# Inference configuration
motion_magnitude = hparams.get("motion_magnitude", False)
window_size = hparams.get("window_size", 30)
input_dim = 6 if motion_magnitude else 5
label_map = {int(k): v for k, v in hparams.get("label_map", {}).items()}
output_dim = len(label_map)
hidden_dim = hparams.get("hidden_dim", 64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# API input/output
class SleepInput(BaseModel):
    model_type: str
    heart_rate: list[float]
    motion_x: list[float]
    motion_y: list[float]
    motion_z: list[float]
    steps: list[float]
    motion_magnitude: list[float] = []

class SleepOutput(BaseModel):
    predicted_label: str

# Initialize FastAPI
app = FastAPI()

# Model setup
def load_model(model_type: str, input_dim: int, output_dim: int, device: torch.device):
    if model_type == "LSTM":
        model = SleepStageLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        model_path = "inference/production_models/LSTM_best_model.pt"
    elif model_type == "GRU":
        model = SleepStageGRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        model_path = "inference/production_models/GRU_best_model.pt"
    elif model_type == "RNN":
        model = SleepStageRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        model_path = "inference/production_models/RNN_best_model.pt"
    elif model_type == "Transformer":
        model = SleepStageRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        model_path = "inference/production_models/Transformer_best_model.pt"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@app.post("/predict", response_model=SleepOutput)
def predict(data: SleepInput):
    try:
        # Build feature matrix
        feature_list = [
            data.heart_rate,
            data.motion_x,
            data.motion_y,
            data.motion_z,
            data.steps
        ]

        if motion_magnitude:
            if not data.motion_magnitude:
                raise HTTPException(status_code=400, detail="motion_magnitude is enabled but missing.")
            feature_list.append(data.motion_magnitude)

        features = np.stack(feature_list, axis=1)

        if len(features) < window_size:
            raise HTTPException(status_code=400, detail=f"Sequence must be at least {window_size} time steps long.")

        features = features[-window_size:]  # Take only the latest window

        # Apply the saved scaler
        features_scaled = scaler.transform(features)
        x = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        # Load model dynamically
        model = load_model(data.model_type, input_dim=input_dim, output_dim=output_dim, device=device)

        with torch.no_grad():
            output = model(x)
            pred_idx = output.argmax(dim=1).item()
        predicted_label = label_map.get(pred_idx, str(pred_idx))
        return {"predicted_label": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


