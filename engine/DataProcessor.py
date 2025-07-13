import sys

import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from engine.SleepDataset import SleepDataset
import torch

class PSGDataProcessor:
    def __init__(self, DATASET_DIR, negative_times=False, run_folder=None, data_alignment=None, motion_magnitude=False, window_size=30, stride=15, label_map=None):
        print("Loading data...")
        self.DATASET_DIR = DATASET_DIR
        self.negative_times = negative_times
        self.run_folder = run_folder
        self.data_alignment = data_alignment
        self.motion_magnitude = motion_magnitude
        self.log_path = os.path.join(self.run_folder, "data_check_failures.txt")
        self.window_size = window_size
        self.stride = stride
        self.label_map = label_map

    def load_file(self, path=None, columns=None):
        filename = os.path.basename(path)

        # Set the correct separator for heart rate and steps
        if "heartrate" in filename or "steps" in filename:
            sep = ','
        else:
            sep = ' '
        df = pd.read_csv(path, sep=sep, names=columns, engine='python')

        # Assure "time" column does not contain strings or NaN
        df["time"] = pd.to_numeric(df["time"], errors='coerce').astype(float)
        df.dropna(subset=["time"], inplace=True)

        if not self.negative_times:
            # Remove negative timestamps as labels are defined from t=0 onwards (PSG start)
            df = df[df["time"] >= 0]

        return df.sort_values("time")


    def sync_data(self, subject_id):

        # Load all signals
        hr = self.load_file(f"{self.DATASET_DIR}/heart_rate/{subject_id}_heartrate.txt", ["time", "hr"])
        motion = self.load_file(f"{self.DATASET_DIR}/motion/{subject_id}_acceleration.txt", ["time", "x", "y", "z"])
        steps = self.load_file(f"{self.DATASET_DIR}/steps/{subject_id}_steps.txt", ["time", "steps"])
        labels = self.load_file(f"{self.DATASET_DIR}/labels/{subject_id}_labeled_sleep.txt", ["time", "label"])

        # Log empty sources
        empty_sources = []
        for name, df in zip(["heart_rate", "motion", "steps", "labels"], [hr, motion, steps, labels]):
            if df.empty:
                empty_sources.append(name)
        if empty_sources:
            with open(self.log_path, "a") as log_file:
                log_file.write(f"Subject {subject_id} failed: empty sources: {', '.join(empty_sources)}\n")
            return None

        # Determine a valid overlapping window
        min_time = max(df["time"].min() for df in [hr, motion, steps, labels])
        max_time = min(df["time"].max() for df in [hr, motion, steps, labels])
        if max_time - min_time < 60:
            with open(self.log_path, "a") as log_file:
                log_file.write(f"Subject {subject_id} failed: insufficient overlapping data\n")
            return None

        # Snap labels to 30-second bins if needed
        if self.data_alignment == 30:
            labels["time"] = labels["time"] - (labels["time"] % 30)
            labels = labels.drop_duplicates("time")

        # Build timeline based on data_alignment
        step = 1 if self.data_alignment is None else self.data_alignment
        time_base = pd.DataFrame({"time": np.arange(min_time, max_time + 1, step)})

        # Merge signals
        df = pd.merge_asof(time_base, hr, on="time", direction='nearest', tolerance=10)
        df = pd.merge_asof(df, motion, on="time", direction='nearest', tolerance=1)
        df = pd.merge_asof(df, steps, on="time", direction='nearest', tolerance=60)
        df = pd.merge_asof(df, labels, on="time", direction='nearest', tolerance=30)

        # Add motion magnitude
        if self.motion_magnitude:
            df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

        # Drop incomplete rows
        df.dropna(inplace=True)

        if df.empty:
            return None

        return df

    def load_all_subjects(self):
        all_samples = []
        all_labels = []
        subject_paths = glob(os.path.join(self.DATASET_DIR, "heart_rate", "*_heartrate.txt"))
        subject_ids = [os.path.basename(p).split("_")[0] for p in subject_paths]
        total_ids = len(subject_ids)
        print(f"Found {total_ids} subjects.\n")
        scaler = StandardScaler()

        # Load and aggregate data from all subjects
        for subject_id in tqdm(subject_ids, desc="Processing subjects"):
            df = self.sync_data(subject_id)
            if df is None:
                total_ids -= 1
                continue
            dataset = SleepDataset(df, self.window_size, self.stride, scaler=scaler, motion_magnitude=self.motion_magnitude, run_path=self.run_folder)
            for i in range(len(dataset)):
                X, y = dataset[i]
                all_samples.append(X)
                all_labels.append(y)
        print(f'Using data from {total_ids} subjects, see sync error logs {self.log_path}\n')
        all_samples = torch.stack(all_samples)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        full_dataset = torch.utils.data.TensorDataset(all_samples, all_labels)

        return full_dataset


