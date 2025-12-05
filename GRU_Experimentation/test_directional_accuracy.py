import torch
import numpy as np
import pandas as pd
from GRU import (
    GRU, LSTM, EncoderGRU,
    GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, NUM_GRU_LAYERS,
    INIT_LEARNING_RATE, NUM_FEATURES,
    NPZ_PATH, COL_NAMES_PATH, SAVE_PATH, MODEL_TO_USE
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
data = np.load(NPZ_PATH)
col_names = pd.read_csv(COL_NAMES_PATH, header=None).squeeze().tolist()
close_idx = col_names.index("log_return_close")

X_test_batches = torch.from_numpy(data["X_test_batches"]).float().to(device)
Y_test_batches = torch.from_numpy(data["Y_test_batches"]).float().to(device)

# Load normalization stats
mu_sigma_df = pd.read_csv("../Data Processing/mu_sigma_df.csv")
mu = mu_sigma_df["mu"].to_numpy()
sigma = mu_sigma_df["sigma"].to_numpy()

# Load model
if MODEL_TO_USE == 'GRU':
    model = GRU(
        input_size=NUM_FEATURES,
        latent_size=32,
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=GRU_OUTPUT_SIZE,
        num_layers=NUM_GRU_LAYERS,
    )
elif MODEL_TO_USE == 'LSTM':
    model = LSTM(
        input_size=NUM_FEATURES,
        latent_size=32,
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=GRU_OUTPUT_SIZE,
        num_layers=NUM_GRU_LAYERS,
    )
elif MODEL_TO_USE == 'EncoderGRU':
    model = EncoderGRU(
        input_size=NUM_FEATURES,
        latent_size=32,
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=GRU_OUTPUT_SIZE,
        num_layers=NUM_GRU_LAYERS,
    )

checkpoint = torch.load(SAVE_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Compute directional accuracy and R^2
correct = 0
total = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in zip(X_test_batches, Y_test_batches):
        # Get residual prediction
        pr_resid = model(xb)

        # Add last close value to get full prediction (same as in GRU.py)
        last = xb[:, -1, close_idx].unsqueeze(-1)
        pr = pr_resid + last

        # Denormalize predictions and targets (same as in GRU.py)
        pr_raw = pr[:, 0] * sigma[close_idx] + mu[close_idx]
        yb_raw = yb[:, 0] * sigma[close_idx] + mu[close_idx]

        # Compute directional accuracy
        pred_sign = torch.sign(pr_raw)
        true_sign = torch.sign(yb_raw)
        mask = (true_sign != 0)
        correct += int((pred_sign[mask] == true_sign[mask]).sum().item())
        total += int(mask.sum().item())

        # Collect for R^2
        all_preds.append(pr_raw)
        all_targets.append(yb_raw)

# Calculate directional accuracy
directional_accuracy = (correct / total) * 100 if total > 0 else 0

# Calculate R^2
all_preds = torch.cat(all_preds).cpu().numpy()
all_targets = torch.cat(all_targets).cpu().numpy()
ss_res = np.sum((all_targets - all_preds) ** 2)
ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

print(f"\n{'='*60}")
print(f"Test Set Directional Accuracy: {directional_accuracy:.2f}% ({correct}/{total})")
print(f"Test Set R²: {r2:.4f}")
print(f"{'='*60}")
