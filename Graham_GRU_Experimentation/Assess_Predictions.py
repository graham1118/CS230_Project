import torch
import numpy as np
import pandas as pd
import sys
from GRU import (
    GRU, LSTM, EncoderGRU,
    GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, NUM_GRU_LAYERS,
    INIT_LEARNING_RATE, NUM_FEATURES,
    NPZ_PATH, COL_NAMES_PATH, SAVE_PATH
)

# ============================================================
# CONFIG
# ============================================================
MODEL = 'EncoderGRU' #or 'GRU' or 'LSTM'
PRINT_RAW = False          # if False, prints normalized/denoised values
PRINT_LAST_N_POINTS = 6
SEQ_LEN = 64
SCALE_Y = 1000.0           # Y was multiplied by 1000 before normalization

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# Load Preprocessed Data
# ============================================================
data = np.load(NPZ_PATH)
col_names = pd.read_csv(COL_NAMES_PATH, header=None).squeeze().tolist()
close_idx = col_names.index("log_return_close")

X_test_batches = torch.from_numpy(data["X_test_batches"]).float().to(device)
Y_test_batches = torch.from_numpy(data["Y_test_batches"]).float().to(device)

X_train_batches = data["X_train_batches"]
Y_train_batches = data["Y_train_batches"]

# ============================================================
# Load Saved Model
# ============================================================

if MODEL == 'GRU':
    model = GRU(
        input_size=NUM_FEATURES,   # 10
        latent_size=32,            # must match the trained encoder output
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=GRU_OUTPUT_SIZE,
        num_layers=NUM_GRU_LAYERS,
    )
elif MODEL == 'LSTM':
    model = LSTM(
        input_size=NUM_FEATURES,   # 10
        latent_size=32,            # must match the trained encoder output
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=GRU_OUTPUT_SIZE,
        num_layers=NUM_GRU_LAYERS,
    )
elif MODEL == 'EncoderGRU':
    model = EncoderGRU(
        input_size=NUM_FEATURES,   # 10
        latent_size=32,            # must match the trained encoder output
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=GRU_OUTPUT_SIZE,
        num_layers=NUM_GRU_LAYERS,
    )
optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")


### Loadin gin Model
checkpoint = torch.load(SAVE_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
if checkpoint["scheduler_state_dict"] is not None:
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

epoch = checkpoint["epoch"]
val_loss = checkpoint["val_loss"]
print(f"Loaded model from epoch {epoch}, val_loss={val_loss:.4f}")

model.to(device)
model.eval()

# ============================================================
# Load mu/sigma for X and compute mu_y/sig_y for Y
# ============================================================
mu_sigma_df = pd.read_csv("../Data Processing/mu_sigma_df.csv")
mu = mu_sigma_df["mu"].to_numpy()
sigma = mu_sigma_df["sigma"].to_numpy()

# Y stats — must be computed from training Ys
Y_train_flat = Y_train_batches.reshape(-1, 1)
mu_y = float(Y_train_flat.mean())
sig_y = float(Y_train_flat.std(ddof=0) + 1e-8)

# ============================================================
# Make Predictions
# ============================================================
predictions = []
with torch.no_grad():
    for xb in X_test_batches:
        xb = xb.to(device).float()
        pr = model(xb)
        predictions.append(pr.cpu())

predictions = torch.cat(predictions, dim=0).numpy()

# ============================================================
# --- Correct Postprocessing ---
# ============================================================

# --- Process Xs (for sequence display) ---
X_test_flat = X_test_batches.cpu().numpy().reshape(-1, SEQ_LEN, X_test_batches.shape[-1])
actual_sequences_raw = X_test_flat[:, :, close_idx]
actual_sequences = actual_sequences_raw * sigma[close_idx] + mu[close_idx]

# --- Process Ys (actual returns) ---
Y_test_flat = Y_test_batches.cpu().numpy().reshape(-1, 1)
actual_returns = (Y_test_flat * sig_y + mu_y) / SCALE_Y

# --- Process Predictions ---
pred_flat = predictions.reshape(-1, 1)
pred_returns = (pred_flat * sig_y + mu_y) / SCALE_Y

print(f"✅ Shapes aligned:")
print(f"  Sequences: {actual_sequences.shape}")
print(f"  Actual returns: {actual_returns.shape}")
print(f"  Predicted returns: {pred_returns.shape}\n")

# ============================================================
# Display Loop
# ============================================================
idx = np.random.randint(0, high=actual_sequences.shape[0] - 100)

while True:
    
    if PRINT_RAW:
        seq = actual_sequences_raw[idx, :].tolist()
        actual_str = f"{actual_returns[idx].item() * SCALE_Y:>10.2f} (actual)"
        pred_str = f"{pred_returns[idx].item() * SCALE_Y:>10.2f} (predicted)"
    else:
        seq = actual_sequences[idx, :].tolist()
        actual_str = f"{actual_returns[idx].item():>10.3f} (actual)"
        pred_str = f"{pred_returns[idx].item():>10.3f} (predicted)"

    seq_tail = [round(x, 3) for x in seq[-PRINT_LAST_N_POINTS:]]
    print(f"Sequence: {seq_tail} --> {actual_str}")
    print(f"{' ' * (len('Sequence: ') + len(str(seq_tail)) + 5)}{pred_str}")
    x = input("View next prediction? y/n? ")
    print("\n")

    if x.lower() != "y":
        sys.exit(0)

    idx += 1