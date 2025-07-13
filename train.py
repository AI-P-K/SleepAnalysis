import argparse
import os
os.environ["MPLCONFIGDIR"] = "/app/runs"
import sys
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from engine.DataProcessor import PSGDataProcessor
from engine.Architectures import *


parser = argparse.ArgumentParser(description="Train sleep stage classification model")


parser.add_argument('--dataset_dir', type=str, required=True,
                    help='Path to the root directory of the dataset')

# Data processor hyperparameters
parser.add_argument('--negative_times', action='store_true',
                    help='Whether to keep or not negative timestamps. By default negative timestamps are kept.')

parser.add_argument('--data_alignment', type=int, default=1,
                    help='Offset in seconds to align sensor timestamps with label timestamps, use 30 for clinical alignment')

parser.add_argument('--motion_magnitude', action='store_true',
                    help='Whether to add or not motion magnitude feature to the dataset')

parser.add_argument('--window_size', type=int, default=30,
                    help='Length of the time window (in seconds) used to segment input sequences for training and inference.')

parser.add_argument('--stride', type=int, default=15,
                    help='Step size (in seconds) to move the window when generating input sequences. Smaller values increase overlap between windows.')


# Training hyperparameters
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for model training (default: 32)')

parser.add_argument('--hidden_dim', type=int, default=64,
                    help='Number of hidden units')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate for model training (default: 1e-3)')

parser.add_argument('--epochs', type=int, default=200,
                    help='Epochs for model training (default: 200)')

parser.add_argument('--train_split', type=float, default=0.9,
                    help='Train/val split (default: 0.9) 90% of data for training 10% for validation')

args = parser.parse_args()


LABEL_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}
NUM_CLASSES = len(set(LABEL_MAP.values()))
generator = torch.Generator().manual_seed(42)
def main():

    def create_run_folder(base_dir="runs"):
        today_str = datetime.now().strftime("%d.%m.%Y")
        day_path = os.path.join(base_dir, today_str)
        os.makedirs(day_path, exist_ok=True)

        # Find next available run_N directory
        run_id = 1
        while True:
            run_path = os.path.join(day_path, f"run_{run_id}")
            if not os.path.exists(run_path):
                os.makedirs(run_path)
                return run_path
            run_id += 1

    run_folder = create_run_folder()
    print(f"Logging this run to: {run_folder}")

    hyperparams = vars(args)
    hyperparams["label_map"] = LABEL_MAP

    hparams_path = os.path.join(run_folder, "hparams.json")
    with open(hparams_path, "w") as f:
        json.dump(hyperparams, f, indent=4)

    data_processor = PSGDataProcessor(DATASET_DIR=args.dataset_dir,
                                      negative_times=args.negative_times,
                                      run_folder=run_folder,
                                      data_alignment=args.data_alignment,
                                      motion_magnitude=args.motion_magnitude,
                                      window_size=args.window_size,
                                      stride=args.stride,
                                      label_map=LABEL_MAP)

    full_dataset = data_processor.load_all_subjects()

    # Dynamic input dimensions
    sample_X, _ = full_dataset[0]
    input_dim = sample_X.shape[1]

    if args.motion_magnitude:
        df = pd.DataFrame(sample_X.numpy(), columns=["hr", "x", "y", "z", "steps", "magnitude"])
    else:
        df = pd.DataFrame(sample_X.numpy(), columns=["heart_rate", "x", "y", "z", "steps"])

    df.to_csv(f"{run_folder}/validation_sample.csv", index=False)

    train_len = int(args.train_split * len(full_dataset))
    train_set, val_set = random_split(full_dataset, [train_len, len(full_dataset) - train_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train_model(model, dataloader, val_loader, device, model_key=None):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_acc = 0.0
        best_model_state = None
        accuracies = []
        for epoch in tqdm(range(args.epochs), desc=f'Training {model_key} model...') :
            model.train()
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = loss_fn(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate on validation set
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    preds = model(X).argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            accuracies.append(acc)
            # print(f"Epoch {epoch + 1}: Validation Accuracy = {acc:.4f}")

            # Save best model
            if acc > best_acc:
                best_acc = acc
                best_model_state = model.state_dict()
                torch.save(best_model_state, f"{run_folder}/{model_key}_best_model.pt")
                # print(f"✅ New best model saved with accuracy {acc:.4f}")

        print(f"\n🏁 Best validation accuracy achieved: {best_acc:.4f}")
        return accuracies

    all_accuracies = {}
    model = SleepStageLSTM(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=NUM_CLASSES).to(device)
    accuracies = train_model(model, train_loader, val_loader, device, model_key="LSTM")
    all_accuracies["LSTM"] = accuracies

    model = SleepStageGRU(input_dim, hidden_dim=args.hidden_dim, output_dim=NUM_CLASSES).to(device)
    accuracies = train_model(model, train_loader, val_loader, device, model_key="GRU")
    all_accuracies["GRU"] = accuracies

    model = SleepStageRNN(input_dim, hidden_dim=args.hidden_dim, output_dim=NUM_CLASSES).to(device)
    accuracies = train_model(model, train_loader, val_loader, device, model_key="RNN")
    all_accuracies["RNN"] = accuracies

    model = SleepStageTransformer(input_dim, hidden_dim=args.hidden_dim, output_dim=NUM_CLASSES).to(device)
    accuracies = train_model(model, train_loader, val_loader, device, model_key="Transformer")
    all_accuracies["Transformer"] = accuracies

    def save_accuracies(run_folder, all_accuracies):
        """
        Save model accuracies to CSV and plot them with best accuracies in the legend.

        Args:
            run_folder (str): Folder where results should be saved
            all_accuracies (dict): Dictionary of model_key -> list of accuracies per epoch
        """
        os.makedirs(run_folder, exist_ok=True)

        # Save to CSV
        df = pd.DataFrame({k: pd.Series(v) for k, v in all_accuracies.items()})
        csv_path = os.path.join(run_folder, "model_accuracies.csv")
        df.to_csv(csv_path, index_label="epoch")

        # Plotting
        plt.figure(figsize=(10, 6))
        for model_key, acc_list in all_accuracies.items():
            best_acc = max(acc_list)
            plt.plot(range(1, len(acc_list) + 1), acc_list, label=f"{model_key} (Best: {best_acc:.2%})")

        plt.title("Validation Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(run_folder, "model_accuracies_plot.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Accuracies saved to {csv_path}")
        print(f"Plot saved to {plot_path}")


    save_accuracies(run_folder, all_accuracies)

if __name__ == "__main__":
    main()


