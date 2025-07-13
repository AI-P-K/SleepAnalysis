from torch.utils.data import Dataset
import torch
import joblib

class SleepDataset(Dataset):
    def __init__(self, dataframe, window_size, stride, scaler=None, motion_magnitude=None, LABEL_MAP={0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}, run_path=None):
        self.samples = []
        self.labels = []
        self.scaler = scaler

        if motion_magnitude:
            features = dataframe[["hr", "x", "y", "z", "steps", "magnitude"]].values
        else:
            features = dataframe[["hr", "x", "y", "z", "steps"]].values

        labels = dataframe["label"].values
        features = self.scaler.fit_transform(features)
        joblib.dump(scaler, f"{run_path}/scaler.pkl")

        for start in range(0, len(features) - window_size, stride):
            end = start + window_size
            X = features[start:end]
            y = labels[end - 1]  # Use label at end of window
            if y in LABEL_MAP:  # Filter out invalid classes
                self.samples.append(X)
                self.labels.append(list(LABEL_MAP.keys()).index(y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)