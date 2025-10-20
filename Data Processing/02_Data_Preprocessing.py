import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import matplotlib.pyplot as plt

### Parameters ###
sequence_length = 32
train_frac, val_frac, test_frac = 0.9, 0.05, 0.05
batch_size = 256  # must be power of 2 for efficient training
HORIZON = 2
EPS = 1e-8
TA_LENGTH = 14 #technical indicator window for TA library functions
# ROLL_NORM_WINDOW = 50000  # only used for rolling normalization, which proves not to work well






# ============================================================
# 1) Load data from CSV
# ============================================================

df = pd.read_csv('BTCUSDT_30m_10years.csv')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

print(f"Loaded {len(df)} rows from CSV")
print(f"Date range: {df['close_time'].min().strftime('%m/%d/%Y')} to {df['close_time'].max().strftime('%m/%d/%Y')}")  #print data range in MM/DD/YYYY format
print(f"Original Columns: {df.columns.tolist()}\n\n")




# ============================================================
# 2) Add New Technical Indicators and compute log returns
# ============================================================

# Bollander Band Width and Position
df['BB_width'] =  (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0']) / (df['BBM_20_2.0_2.0'] + 1e-8)
df['BB_position'] = (df['close'] - df['BBL_20_2.0_2.0']) / (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0'] + 1e-8)

#MACD 
macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
df = pd.concat([df, macd], axis=1)

#Standard Moving Average
df["SMA_14"] = ta.sma(df["close"], length=14)  

# Taking returns centers the timeseries around 1. Taking log returns centers around 0 and improves stability of training.
log_returns = np.log(df.iloc[:, :4] / df.iloc[:, :4].shift(1)) # First 4 columns only (e.g., open, high, low, close)
log_returns = log_returns.fillna(0)
df.iloc[:,:4] = log_returns
df.columns = ['log_return_open', 'log_return_high', 'log_return_low', 'log_return_close'] + list(df.columns[4:])



### REMOVE UN-NEEDED COLUMNS ###
close_times = df['close_time']
close_times.to_csv('datapoint_timestamps.csv', index=True, header=True)
df.columns.to_series().to_csv("column_names.csv", index=False, header=False)

df.drop(columns=['close_time', 'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0'], inplace=True)
df.drop(columns=['number_of_trades', 'MACD_12_26_9', 'MACDs_12_26_9'], inplace=True)

df = df.iloc[33:,:] # ONLY use FOR MACD
arr = df.to_numpy()

# print("stopping here to check data shapes... can comment out sys.exit(0) to continue to training")
# sys.exit(0)
print("\n#####################################################\n")



# ============================================================
# 3) Split into train/val/test
# ============================================================
#We do this before sequencing to ensure that there is no data leakage
train_size = int(len(arr) * train_frac)
val_size = int(len(arr) * val_frac)
test_size = len(arr) - train_size - val_size

train_set = arr[:train_size]
val_set = arr[train_size:train_size + val_size]
test_set = arr[train_size + val_size:]


# ============================================================
# 4) Fit normalization stats on training set only
# ============================================================

# compute mean and std per feature (axis=0 → column-wise)
mu = np.mean(train_set, axis=0)
sigma = np.std(train_set, axis=0) + EPS  # avoid divide-by-zero

mu_sigma = {"mu": mu, "sigma": sigma}
mu_sigma_df = pd.DataFrame(mu_sigma)
mu_sigma_df.to_csv("mu_sigma_df.csv", header=True)

print("Train raw stds:", np.std(train_set, axis=0))
print("Val raw stds:", np.std(val_set, axis=0))
print("Test raw stds:", np.std(test_set, axis=0))

def normalize(x, mu, sigma):
    """Apply training-set normalization to a NumPy array."""
    return (x - mu) / sigma

train_set = normalize(train_set, mu, sigma)
val_set   = normalize(val_set, mu, sigma)
test_set  = normalize(test_set, mu, sigma)


# ============================================================
# 5) Verify normalization correctness
# ============================================================

print("current_columns", df.columns.tolist())
print("number of features:", train_set.shape[1])
print("Train means:", [f"{x:.3f}" for x in np.mean(train_set, axis=0)])
print("Train stds: ", [f"{x:.3f}" for x in np.std(train_set, axis=0)])
print("Val means:  ", [f"{x:.3f}" for x in np.mean(val_set, axis=0)])
print("Val stds:   ", [f"{x:.3f}" for x in np.std(val_set, axis=0)])
print("Test means: ", [f"{x:.3f}" for x in np.mean(test_set, axis=0)])
print("Test stds:  ", [f"{x:.3f}" for x in np.std(test_set, axis=0)])
print('\n\n')





# ============================================================
# 6) Analyze variance of signals
# ============================================================

# # === Parameters ===
# WINDOW = 1000  # smaller window (~21 days for 30-min data)
# FEATURES = ['log_return_close', 'volume', 'number_of_trades', 'BB_width']
# CSV_PATH = "BTCUSDT_30m_10years.csv"

# # === Load data ===
# df = pd.read_csv(CSV_PATH)
# df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
# df = df.sort_values('close_time')

# # === Compute Bollinger Band width (if not in dataset) ===
# if 'BB_width' not in df.columns:
#     df['BB_width'] = (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0']) / (df['BBM_20_2.0_2.0'] + 1e-8)

# # === Compute log returns ===
# for col in ['open', 'high', 'low', 'close']:
#     df[f'log_return_{col}'] = np.log(df[col] / df[col].shift(1))
# df.fillna(0, inplace=True)

# # === Rolling standard deviation ===
# vol_df = df[[f for f in FEATURES if f in df.columns]].rolling(WINDOW).std()

# # Drop NaNs that appear at the start
# vol_df = vol_df.dropna()
# df = df.loc[vol_df.index]

# # === Train/Val/Test split markers ===
# train_frac, val_frac, test_frac = 0.9, 0.05, 0.05
# n = len(df)
# train_end = int(train_frac * n)
# val_end = int((train_frac + val_frac) * n)
# split_dates = [df.iloc[train_end]['close_time'], df.iloc[val_end]['close_time']]

# # === Plot ===
# plt.figure(figsize=(12, 6))
# for f in vol_df.columns:
#     plt.plot(df['close_time'], vol_df[f], label=f)

# for i, split_date in enumerate(split_dates):
#     plt.axvline(split_date, color='red', linestyle='--', alpha=0.7)
#     plt.text(split_date, plt.ylim()[1]*0.9, ['Train/Val', 'Val/Test'][i],
#              rotation=90, color='red', ha='right', va='top')

# plt.title(f"Rolling Volatility (Window={WINDOW} samples ≈ {WINDOW * 30 / 60 / 24:.1f} days)")
# plt.xlabel("Date")
# plt.ylabel("Rolling Std (Volatility)")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()




# ============================================================
# 7) Sequence Data with no-overlap between sets
# ============================================================

def sequence(data, window, HORIZON):
    """Return list of (X, Y) for all overlapping sequences of length `window`."""
    T = data.shape[0]
    starts = np.arange(0, T - window - HORIZON)
    X = np.stack([data[i:i+window] for i in starts], axis=0)
    #Y = np.stack([data[i+window + HORIZON] - data[i + window - 1] for i in starts], axis=0) #for z-scored OHLC
    Y = np.stack([data[i+window + HORIZON] for i in starts], axis=0) #for log returns
    return X, Y

print("Starting Sequencing...")
X_train, Y_train = sequence(train_set, sequence_length, HORIZON)
X_val, Y_val = sequence(val_set, sequence_length, HORIZON)
X_test, Y_test = sequence(test_set, sequence_length, HORIZON)
print("Sequencing completed")

print(f"Number of examples in train/val/test = {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
print(f"Train set fraction: {X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]):.3f}")
print(f"Val set fraction: {X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]):.3f}")
print(f"Test set fraction: {X_test.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]):.3f}")
print("Each row of X is 1 input sequence of length:", X_train.shape[1])
print("\n#####################################################\n")






# ============================================================
# 8) Create Batches of size `batch_size`
# ============================================================

def create_batches(X, Y, batch_size):
    """Return list of (X_batch, Y_batch) tuples."""
    n = X.shape[0]
    X_batches = []
    Y_batches = []
    Last_X_batch = None
    Last_Y_batch = None
    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batches.append(X[start:end])
        Y_batches.append(Y[start:end])

    #we must store the last batch separately, because it may be smaller than batch_size, and so numpy can't convert it to an array
    Last_X_batch = X_batches[-1] 
    Last_Y_batch = Y_batches[-1]
    X_batches.pop()
    Y_batches.pop()
    
    return np.array(X_batches), np.array(Y_batches), np.array(Last_X_batch), np.array(Last_Y_batch)


#Keep batches as python lists of arrays, so that the last batch can be smaller than batch_size if needed
print("starting batching...")
X_train_batches, Y_train_batches, Last_X_train_batch, Last_Y_train_batch = create_batches(X_train, Y_train, batch_size)
X_val_batches, Y_val_batches, Last_X_val_batch, Last_Y_val_batch = create_batches(X_val, Y_val, batch_size)
X_test_batches, Y_test_batches, Last_X_test_batch, Last_Y_test_batch = create_batches(X_test, Y_test, batch_size)

#Print the number of batches in each set, the size of each batch, and the shape of each example
print(f"train/val/test num batches = {len(X_train_batches)}/{len(X_val_batches)}/{len(X_test_batches)}")
print(f"batch size = {batch_size}")

#print the shape of each batch set
print(f"X_train_batches shape: {X_train_batches[0].shape}, Y_train_batches shape: {Y_train_batches[0].shape}")
print(f"X_val_batches shape: {X_val_batches[0].shape}, Y_val_batches shape: {Y_val_batches[0].shape}")
print(f"X_test_batches shape: {X_test_batches[0].shape}, Y_test_batches shape: {Y_test_batches[0].shape}")
print(f"Size of last batch in train/val/test = {Last_X_train_batch.shape[0]}/{Last_X_val_batch.shape[0]}/{Last_Y_test_batch.shape[0]}")
print(f"Size of last batch")
print("\n#####################################################\n")



assert not np.array_equal(X_train[0, -1], Y_train[0]), "Off-by-one leak!"

#### ---- 7) Save preprocessed data to .npz file ---- ####
np.savez_compressed('preprocessed_data.npz',
    X_train_batches=X_train_batches, Y_train_batches=Y_train_batches,
    Last_X_train_batch=Last_X_train_batch, Last_Y_train_batch=Last_Y_train_batch,
    X_val_batches=X_val_batches, Y_val_batches=Y_val_batches,
    Last_X_val_batch=Last_X_val_batch, Last_Y_val_batch=Last_Y_val_batch,
    X_test_batches=X_test_batches, Y_test_batches=Y_test_batches,
    Last_X_test_batch=Last_X_test_batch, Last_Y_test_batch=Last_Y_test_batch,
)   