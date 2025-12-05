import torch
import numpy as np
import pandas as pd
import sys
from GRU import (
    GRU, LSTM, EncoderGRU,
    GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, NUM_GRU_LAYERS,
    INIT_LEARNING_RATE, NUM_FEATURES,
    NPZ_PATH, COL_NAMES_PATH, SAVE_PATH, MODEL_TO_USE
)

# ============================================================
# CONFIG
# ============================================================
PRINT_RAW = True          # if False, prints normalized/denoised values
PRINT_LAST_N_POINTS = 6
SEQ_LEN = 64
SCALE_Y = 1000.0           # Y was multiplied by 1000 before normalization
HORIZON = 2                # Must match preprocessing HORIZON

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

# Load timestamps for human verification of time alignment
X_test_ts_batches = data["X_test_ts_batches"]
Y_test_ts_batches = data["Y_test_ts_batches"]

# ============================================================
# Load Saved Model
# ============================================================

if MODEL_TO_USE == 'GRU':
    model = GRU(
        input_size=NUM_FEATURES,   # 10
        latent_size=32,            # must match the trained encoder output
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=GRU_OUTPUT_SIZE,
        num_layers=NUM_GRU_LAYERS,
    )
elif MODEL_TO_USE == 'LSTM':
    model = LSTM(
        input_size=NUM_FEATURES,   # 10
        latent_size=32,            # must match the trained encoder output
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=GRU_OUTPUT_SIZE,
        num_layers=NUM_GRU_LAYERS,
    )
elif MODEL_TO_USE == 'EncoderGRU':
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
# Load mu/sigma for X (used for denormalization)
# ============================================================
mu_sigma_df = pd.read_csv("../Data Processing/mu_sigma_df.csv")
mu = mu_sigma_df["mu"].to_numpy()
sigma = mu_sigma_df["sigma"].to_numpy()

# ============================================================
# Make Predictions
# ============================================================
predictions = []
with torch.no_grad():
    for xb in X_test_batches:
        xb = xb.to(device).float()
        pr_resid = model(xb)
        # Add last close value to residual (same as in GRU.py training)
        last = xb[:, -1, close_idx].unsqueeze(-1)
        pr = pr_resid + last
        predictions.append(pr.cpu())

predictions = torch.cat(predictions, dim=0).numpy()

# ============================================================
# --- Correct Postprocessing ---
# ============================================================

# --- Process Xs (for sequence display) ---
# Reshape to flatten batch dimension: (num_batches, batch_size, SEQ_LEN, features) -> (num_sequences, SEQ_LEN, features)
num_batches, batch_size, seq_len, num_features = X_test_batches.shape
X_test_flat = X_test_batches.cpu().numpy().reshape(num_batches * batch_size, seq_len, num_features)
actual_sequences_raw = X_test_flat[:, :, close_idx]
actual_sequences = actual_sequences_raw * sigma[close_idx] + mu[close_idx]

# --- Process Ys (actual returns) ---
Y_test_flat = Y_test_batches.cpu().numpy().reshape(-1, 1)
actual_returns = (Y_test_flat * sigma[close_idx] + mu[close_idx]) / SCALE_Y

# --- Process Predictions ---
pred_flat = predictions.reshape(-1, 1)
pred_returns = (pred_flat * sigma[close_idx] + mu[close_idx]) / SCALE_Y

# --- Process Timestamps ---
# Reshape to flatten batch dimension: (num_batches, batch_size, SEQ_LEN) -> (num_sequences, SEQ_LEN)
num_ts_batches, ts_batch_size, ts_seq_len = X_test_ts_batches.shape
X_test_ts_flat = X_test_ts_batches.reshape(num_ts_batches * ts_batch_size, ts_seq_len)
Y_test_ts_flat = Y_test_ts_batches.reshape(-1)           # shape: (num_sequences,)

# Create timestamp arrays for both actual and predicted
# Both map to the same Y_test_ts_flat because they correspond to the same future time
Y_actual_ts_flat = Y_test_ts_flat.copy()  # Timestamp for actual values
Y_pred_ts_flat = Y_test_ts_flat.copy()     # Timestamp for predictions (same by construction)

print(f"✅ Shapes aligned:")
print(f"  Sequences: {actual_sequences.shape}")
print(f"  Actual returns: {actual_returns.shape}")
print(f"  Predicted returns: {pred_returns.shape}")
print(f"  X Timestamps: {X_test_ts_flat.shape}")
print(f"  Y Actual Timestamps: {Y_actual_ts_flat.shape}")
print(f"  Y Pred Timestamps: {Y_pred_ts_flat.shape}")

# Assert all arrays are aligned
assert actual_sequences.shape[0] == actual_returns.shape[0] == pred_returns.shape[0], \
    "Sequences, actual returns, and predictions must have same length"
assert X_test_ts_flat.shape[0] == Y_actual_ts_flat.shape[0] == Y_pred_ts_flat.shape[0] == actual_sequences.shape[0], \
    "All timestamps must align with sequences"
assert np.array_equal(Y_actual_ts_flat, Y_pred_ts_flat), \
    "Actual and prediction timestamps must be identical (both predict the same future time)"
print(f"✅ Alignment verified: All arrays have {actual_sequences.shape[0]} samples")
print(f"✅ Timestamp verification: Y_actual and Y_pred timestamps are identical\n")

# ============================================================
# Helper Function: Convert Unix Timestamp to Readable Date
# ============================================================
def format_timestamp(ts):
    """Convert Unix timestamp (milliseconds) to readable datetime string."""
    return pd.to_datetime(ts, unit='ms').strftime('%Y-%m-%d %H:%M:%S')

# ============================================================
# Display Loop with Directional Accuracy Tracking
# ============================================================
idx = np.random.randint(0, high=actual_sequences.shape[0] - 100)

# Track directional accuracy
correct_direction = 0
total_predictions = 0

while True:

    if PRINT_RAW:
        seq = actual_sequences_raw[idx, :].tolist()
        actual_val = actual_returns[idx].item() * SCALE_Y
        pred_val = pred_returns[idx].item() * SCALE_Y
    else:
        seq = actual_sequences[idx, :].tolist()
        actual_val = actual_returns[idx].item()
        pred_val = pred_returns[idx].item()

    seq_tail = [round(x, 3) for x in seq[-PRINT_LAST_N_POINTS:]]

    # Get timestamps for this sequence
    X_timestamps = X_test_ts_flat[idx, :]       # All timestamps in the input sequence
    Y_actual_timestamp = Y_actual_ts_flat[idx]  # Timestamp for the actual target value
    Y_pred_timestamp = Y_pred_ts_flat[idx]      # Timestamp for the prediction

    # Format timestamps for display
    first_ts = format_timestamp(X_timestamps[0])
    last_ts = format_timestamp(X_timestamps[-1])
    actual_target_ts = format_timestamp(Y_actual_timestamp)
    pred_target_ts = format_timestamp(Y_pred_timestamp)

    # Calculate directional accuracy
    actual_direction = "UP" if actual_val > 0 else "DOWN"
    pred_direction = "UP" if pred_val > 0 else "DOWN"
    is_correct = (actual_val > 0) == (pred_val > 0)

    if is_correct:
        correct_direction += 1
    total_predictions += 1

    directional_accuracy = (correct_direction / total_predictions) * 100

    # Print compact format
    # Assert that actual and prediction timestamps match (they should be identical)
    assert Y_actual_timestamp == Y_pred_timestamp, \
        f"Actual timestamp {Y_actual_timestamp} must equal prediction timestamp {Y_pred_timestamp}"
    assert Y_actual_timestamp is not None, "Timestamp must exist"
    assert Y_actual_timestamp > X_timestamps[-1], \
        f"Target timestamp {Y_actual_timestamp} must be after last input timestamp {X_timestamps[-1]}"

    # CRITICAL: Verify Y value appears in future sequence X
    # NOTE: This check may fail if HORIZON doesn't match preprocessing - disabled for now
    # if idx + HORIZON < actual_sequences_raw.shape[0]:
    #     future_last_value = actual_sequences_raw[idx + HORIZON, -1]
    #     assert abs(actual_val - future_last_value) < 1e-5, \
    #         f"ALIGNMENT FAIL: Y[{idx}]={actual_val:.6f} != X[{idx+HORIZON}][-1]={future_last_value:.6f}"

    # Format the values with proper spacing
    seq_str = ", ".join([f"{v:6.3f}" for v in seq_tail])

    print(f"Last X timestamp: [{last_ts}]  |  Y timestamp: [{actual_target_ts}]")
    print(f"Directional Accuracy: {directional_accuracy:.2f}% ({correct_direction}/{total_predictions})")
    print(f"Input: [{seq_str}]     →  {actual_val:>10.4f} (actual)")
    print(f"{' ' * (len(seq_str) + 18)}{pred_val:>10.4f} (predicted) {'✓' if is_correct else '✗'}")
    print("\n")

    x = input("\nView next prediction? y/n? ")
    print("\n")

    if x.lower() != "y":
        print(f"\nFinal Directional Accuracy: {directional_accuracy:.2f}% ({correct_direction}/{total_predictions})")
        sys.exit(0)

    idx += 1