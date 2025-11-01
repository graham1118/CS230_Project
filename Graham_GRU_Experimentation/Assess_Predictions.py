import torch
from torch import nn
import numpy as np
import pandas as pd
import sys
from GRU import GRU, LSTM, GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, NUM_GRU_LAYERS, INIT_LEARNING_RATE, NUM_FEATURES
from GRU import NPZ_PATH, COL_NAMES_PATH, SAVE_PATH
from GRU import load_preprocessed_data


HORIZON = 2 #MANUALLY UPDATE
PRINT_LAST_N_POINTS = 6
DISPLAY_RAW = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("USING DEVICE:", device)


def kalman_denoise_multifeature(
    X,
    process_var=1e-5,     # process noise variance (Q)
    obs_var=1e-3          # observation noise variance (R)
):
    """
    Apply a simple 1D Kalman filter to each feature column independently.
    State model: x_t = x_{t-1} + w_t
    Observation: z_t = x_t + v_t

    Parameters
    ----------
    X : np.ndarray
        Shape (n_timesteps, n_features)
    process_var : float
        Process noise variance (how much the hidden state can drift)
    obs_var : float
        Observation noise variance (how noisy measurements are)

    Returns
    -------
    np.ndarray
        Smoothed data of same shape.
    """
    X = np.asarray(X, dtype=np.float32)
    n, m = X.shape
    out = np.zeros_like(X)

    for j in range(m):
        z = X[:, j]
        x_est = np.zeros(n, dtype=np.float32)
        P = 1.0  # initial estimate uncertainty
        x_est[0] = z[0]

        for t in range(1, n):
            # Prediction
            x_pred = x_est[t-1]
            P_pred = P + process_var

            # Kalman Gain
            K = P_pred / (P_pred + obs_var)

            # Update
            x_est[t] = x_pred + K * (z[t] - x_pred)
            P = (1 - K) * P_pred

        out[:, j] = x_est

    return out


def reconstruct_close_from_Y(
    raw_data: pd.DataFrame,
    actual_returns,          # shape (N,) or (N,1), torch or numpy
    mu, sigma,               # vectors from training; mu[3], sigma[3] are for log_return_close
    train_frac=0.90, val_frac=0.05, test_frac=0.05,
    window=32, HORIZON=2,
    macd_skip=33             # you used: df = df.iloc[33:, :]
):
    # 0) Ensure numpy 1D
    if hasattr(actual_returns, "detach"):   # torch tensor -> numpy
        actual_returns = actual_returns.detach().cpu().numpy()
    actual_returns = np.asarray(actual_returns).reshape(-1)

    # 1) De-normalize close log-returns
    r = actual_returns * sigma[3] + mu[3]   # r_t = log(P_t / P_{t-1})
    r = r/1000

    # 2) Compute where the test Y’s live in the raw CSV index space
    N_raw = len(raw_data)
    # After macd_skip, you formed df and then arr = df.to_numpy()
    N_arr = N_raw - macd_skip

    train_size = int(N_arr * train_frac)
    val_size   = int(N_arr * val_frac)
    # test_size = N_arr - train_size - val_size  # not strictly needed

    arr_test_start = train_size + val_size                 # index in arr where test_set starts
    raw_test_start = macd_skip + arr_test_start            # same timestamp in raw_data
    raw_y_start = raw_test_start + window + HORIZON        # first timestamp used in Y_test

    N = len(r)  # number of Y points we have
    # 3) Ground-truth close prices at those timestamps (one per Y)
    true_prices = raw_data["close"].iloc[raw_y_start : raw_y_start + N].to_numpy()

    # 4) Reconstruct prices from returns:
    #    Need the price immediately BEFORE the first Y timestamp as the seed.
    p0 = raw_data["close"].iloc[raw_y_start - 1]
    recon = np.empty(N, dtype=np.float64)
    recon[0] = p0 * np.exp(r[0])
    for t in range(1, N):
        recon[t] = recon[t-1] * np.exp(r[t])

    return recon, true_prices, raw_y_start


def reconstruct_last_block(
    Y_block,
    raw_data,
    mu,
    sigma,
    close_idx,
    scale=1000.0,
    horizon=2,
    macd_skip=33
):
    """
    Reconstruct true close prices for the most recent (last) block.

    Parameters
    ----------
    Y_block : np.ndarray
        Model outputs or Y labels (log-return * scale), shape (num_sequences, 1)
    raw_data : pd.DataFrame
        Original dataframe containing the 'close' column before processing.
    mu, sigma : np.ndarray or pd.Series
        Mean and std used during normalization (from training).
    close_idx : int
        Index of the 'log_return_close' column in your processed data.
    scale : float, optional
        Multiplier used on Y during preprocessing (default 1000.0)
    horizon : int, optional
        Prediction horizon, number of steps ahead (default 2)
    macd_skip : int, optional
        Number of initial rows dropped after computing MACD (default 33)

    Returns
    -------
    recon_prices : np.ndarray
        Reconstructed prices for the last block.
    true_prices : np.ndarray
        Ground-truth close prices for comparison.
    """
    # 1️⃣ Ensure numpy 1D
    Y_block = np.asarray(Y_block).reshape(-1)

    # 2️⃣ Undo normalization and scaling
    r = (Y_block * sigma[close_idx]) + mu[close_idx]   # undo normalization
    r /= scale                                         # undo 1000× scaling if applied
    # r now represents log returns

    # 3️⃣ Determine number of reconstructed points and data alignment
    N = len(r)
    raw_trimmed = raw_data.iloc[macd_skip:]  # reflect your df = df.iloc[33:,:] step

    # 4️⃣ True close prices corresponding to the last N Y values
    true_prices = raw_trimmed["close"].iloc[-N:].to_numpy()

    # 5️⃣ Get starting price (previous close)
    p0 = raw_trimmed["close"].iloc[-N - horizon]

    # 6️⃣ Reconstruct prices iteratively
    recon_prices = np.empty(N, dtype=np.float64)
    recon_prices[0] = p0 * np.exp(r[0])
    for t in range(1, N):
        recon_prices[t] = recon_prices[t - 1] * np.exp(r[t])

    return recon_prices, true_prices



# ============================================================
# Load in Test Data
# ============================================================
data = load_preprocessed_data(path=NPZ_PATH)
feature_names = pd.read_csv(COL_NAMES_PATH)

X_test_batches  = torch.from_numpy(data["X_test_batches"]).to(torch.float32).to(device)
Y_test_batches  = torch.from_numpy(data["Y_test_batches"]).to(torch.float32).to(device)

close_idx = 3
SEQ_LEN = X_test_batches.shape[2]



# ============================================================
# Load saved model
# ============================================================
# Must define model architecture exactly the same way as in GRU.py
model = GRU(input_size=NUM_FEATURES, hidden_size=GRU_HIDDEN_SIZE, output_size=GRU_OUTPUT_SIZE, num_layers=NUM_GRU_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# Load saved state
checkpoint = torch.load(SAVE_PATH, map_location='cpu')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
if checkpoint['scheduler_state_dict'] is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
print(f"Loaded model from epoch {epoch}, val_loss={val_loss:.4f}")




# ============================================================
# Retreive raw prices and verify time alignment
# ============================================================

raw_data = pd.read_csv("../Data Processing/BTCUSDT_30m_10years.csv")
mu_sigma = pd.read_csv("../Data Processing/mu_sigma_df.csv")
mu = mu_sigma['mu']
sigma = mu_sigma['sigma']




# ============================================================
# Make Predictions
# ============================================================

model.eval()
predictions = []
with torch.no_grad():
    for x in X_test_batches:
        x = x.to(device).float()
        pr = model(x)
        predictions.append(pr.cpu())

# Concatenate all predictions
predictions = torch.cat(predictions, dim=0).numpy()




#### Process Xs, Ys, and Predictions (aligned properly) ####
# Flatten across batches while preserving one-to-one alignment
X_test_flat = X_test_batches.reshape(-1, SEQ_LEN, X_test_batches.shape[-1])
Y_test_flat = Y_test_batches.reshape(-1, 1)
pred_flat   = predictions.reshape(-1, 1)

# Extract close feature from X
actual_sequences_raw = X_test_flat[:, :, close_idx].cpu().numpy()
actual_sequences     = actual_sequences_raw * sigma[close_idx] + mu[close_idx]

# Unnormalize Y and predictions (undo normalization, then divide by 1000)
actual_returns_raw = Y_test_flat.cpu().numpy()
actual_returns     = (actual_returns_raw * sigma[close_idx] + mu[close_idx]) / 1000

pred_returns_raw = pred_flat
pred_returns     = (pred_returns_raw * sigma[close_idx] + mu[close_idx]) / 1000

print(actual_sequences.shape)
print(actual_returns.shape)
print(pred_returns.shape)




# ============================================================
# Manually view Sequences and prediction
# ============================================================
while True:
    idx = np.random.randint(0, high=actual_sequences.shape[0])

    if DISPLAY_RAW:
        seq = actual_sequences_raw[idx,:].tolist()
        actual_str = f"{actual_returns_raw[idx].item():>10.2f} (actual)"
        pred_str   = f"{pred_returns_raw[idx].item():>10.2f} (predicted)"
    else:
        seq = actual_sequences[idx,:].tolist()
        actual_str = f"{actual_returns[idx].item():>10.2f} (actual)"
        pred_str   = f"{pred_returns[idx].item():>10.2f} (predicted)"

    print(f"Sequence: {[round(seq[i], 3) for i in range(SEQ_LEN - PRINT_LAST_N_POINTS, SEQ_LEN)]} --> {actual_str}")
    print(f"{' ' * (len('Sequence: ') + len(str([round(seq[i], 3) for i in range(SEQ_LEN - PRINT_LAST_N_POINTS, SEQ_LEN)])) + 5)}{pred_str}")
    x = input("View next prediction? y/n? ")
    print("\n")

    if x == 'y':
        continue
    else:
        sys.exit(0)









# raw_denoised = kalman_denoise_multifeature(raw_data.iloc[:,3])
# #takes the Y labels from the dataset and returns them as recon_prices
# recon_prices, true_prices, start_idx = reconstruct_close_from_Y(
#     raw_denoised, Y_test, mu, sigma,
#     train_frac=0.90, val_frac=0.05, test_frac=0.05,
#     window=SEQ_LEN, HORIZON=HORIZON, macd_skip=33
# )


# print("VERIFYING TIME ALIGNMENT OF TRUE AND RECONSTRUCTED LABEL PRICES")
# print("last 5 reconstructed:", np.round(recon_prices[-5:], 1))
# print("last 5 true:         ", np.round(true_prices[-5:], 1))
# print("This was just a test. If these line up, we can go ahead and turn predicted returns into predicted prices")


# #convert pred_returns into pred_prices
# pred_prices, _, _ = reconstruct_close_from_Y(
#     raw_data, pred_returns, mu, sigma,
#     train_frac=0.90, val_frac=0.05, test_frac=0.05,
#     window=SEQ_LEN, HORIZON=HORIZON, macd_skip=33
# )