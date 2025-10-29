import torch
from torch import nn
import numpy as np
import pandas as pd
from GRU import GRU, LSTM, GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, NUM_GRU_LAYERS, INIT_LEARNING_RATE, NUM_FEATURES
from GRU import NPZ_PATH, COL_NAMES_PATH, SAVE_PATH
from GRU import load_preprocessed_data


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("USING DEVICE:", device)


# ============================================================
# Load in Test Data
# ============================================================
data = load_preprocessed_data(path=NPZ_PATH)
feature_names = pd.read_csv(COL_NAMES_PATH)

X_test_batches = data["X_test_batches"]
Y_test_batches = data["Y_test_batches"]

X_test_batches_torch  = torch.from_numpy(X_test_batches).to(torch.float32).to(device)
Y_test_batches_torch  = torch.from_numpy(Y_test_batches).to(torch.float32).to(device)

num_test_batches = X_test_batches.shape[0]
BATCH_SIZE = X_test_batches.shape[1]  #assumes all batches have same size
SEQ_LEN = X_test_batches.shape[2]
NUM_FEATURES = X_test_batches.shape[3]




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
# Retreive true and predicted CLOSE PRICES ONLY
# ============================================================

model.eval()
predictions = []
with torch.no_grad():
    for xb in X_test_batches_torch:
        xb = xb.to(device).float()
        pr = model(xb)
        predictions.append(pr.cpu())

# Concatenate all predictions
predictions = torch.cat(predictions, dim=0).numpy()
pred_returns = predictions[:, 0]

Y_close = Y_test_batches_torch[:,:,3]          # take only the first feature (close)
actual_returns = Y_close.reshape(-1, 1)  # flatten across batches

print("✅ Pred Close Returns shape:", pred_returns.shape)
print("✅ Actual Close Returns shape:", actual_returns.shape)



mu_sigma_df = pd.read_csv("../Data Processing/mu_sigma_df.csv")
mu = mu_sigma_df["mu"]
sigma = mu_sigma_df["sigma"]
print(mu)
print(sigma)

assert actual_returns.shape[0] == pred_returns.shape[0], "ERROR: actual returns shape != predicted returns shape!!"


# ============================================================
# Retreive raw prices and verify time alignment
# ============================================================

raw_data = pd.read_csv("../Data Processing/BTCUSDT_30m_10years.csv")



#last_n_datapoints = actual_returns.shape[0]

OFFSET = 33 + 32 + 2   # MACD skip + window + horizon
END_TRIM = 2           # final HORIZON rows not used

raw_prices = raw_data["close"].iloc[OFFSET : -END_TRIM]
log_returns2 = np.log(raw_prices / raw_prices.shift(1)).fillna(0)
log_returns2 = (log_returns2 - mu[3]) / sigma[3]
log_returns2_tail = log_returns2.iloc[-len(actual_returns):]

# print(log_returns2[-5:])
# print("#########")
# print(actual_returns[-5:])





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

    # 2) Compute where the test Y’s live in the raw CSV index space
    N_raw = len(raw_data)
    # After macd_skip, you formed df and then arr = df.to_numpy()
    N_arr = N_raw - macd_skip

    train_size = int(N_arr * train_frac)
    val_size   = int(N_arr * val_frac)
    # test_size = N_arr - train_size - val_size  # not strictly needed

    arr_test_start = train_size + val_size                 # index in arr where test_set starts
    raw_test_start = macd_skip + arr_test_start            # same timestamp in raw_data
    # Y_test indices inside test_set start at (window + HORIZON)
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


recon_prices, true_prices, start_idx = reconstruct_close_from_Y(
    raw_data, actual_returns, mu, sigma,
    train_frac=0.90, val_frac=0.05, test_frac=0.05,
    window=32, HORIZON=2, macd_skip=33
)

print("start idx in raw_data:", start_idx)
print("last 5 reconstructed:", np.round(recon_prices[-5:], 5))
print("last 5 true:         ", np.round(true_prices[-5:], 5))

# quick error metrics
abs_pct_err = np.abs(recon_prices - true_prices) / np.maximum(true_prices, 1e-12)
print("MAE:", np.mean(np.abs(recon_prices - true_prices)))
print("MAPE:", 100*np.mean(abs_pct_err), "%")

# ============================================================
# Manually view Sequences and prediction
# ============================================================

#### Aggregate across all batches
X_close = X_test_batches_torch[..., 0]          # take only the first feature (close)
X_close_flat = X_close.reshape(-1, X_close.shape[-1])  # flatten across batches
test_sequences = X_close_flat # (n, 11) matrix, most recent at bottom

