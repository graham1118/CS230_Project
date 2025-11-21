# Data preprocessing file for Attention Model

import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import matplotlib.pyplot as plt
import pywt
from scipy.signal import lfilter, butter

### Parameters ###
SEED = 42
rng = np.random.default_rng(seed=SEED)

#### Can be any combination of True/False ####
COMBINE_DATA = False
BLOCK_SHUFFLE = False
DENOISE = True
BLOCK_SIZE = 2048

T1 = 6          # Validation prefix period
T2 = 10         # Final forecast period after validation
HORIZON_T = 16  # T = T1 + T2 (total decoder length)
ENCODER_LENGTH_P = 32  # Past context window (P)

EPS = 1e-8
TA_LENGTH = 14  # Technical indicator window for TA library functions
SELECT_RANGE = False  # Used if you want to select a later range of the data
USE_MACD = True

train_frac, val_frac, test_frac = 0.9, 0.05, 0.05
batch_size = 128  # Must be power of 2 for efficient training
#############################################

# Verify T = T1 + T2
assert HORIZON_T == T1 + T2, f"HORIZON_T ({HORIZON_T}) must equal T1 + T2 ({T1 + T2})"

#### WAVELET DENOISING ####
def swt_denoise_multifeature(x, wavelet='db4', level=3, thresh_scale=0.7):
    m = 2 ** level
    target = int(np.ceil(len(x) / m) * m)
    pad_len = target - len(x)

    # pad on the right using last value
    if pad_len > 0:
        x_pad = np.pad(x, (0, pad_len), mode='edge')
    else:
        x_pad = x

    coeffs = pywt.swt(x_pad, wavelet=wavelet, level=level)
    d1 = coeffs[-1][1]
    sigma = np.median(np.abs(d1)) / 0.6745
    lam = thresh_scale * sigma * np.sqrt(2 * np.log(len(x_pad)))
    coeffs_t = [(cA, pywt.threshold(cD, lam, mode='soft')) for (cA, cD) in coeffs]
    x_rec = pywt.iswt(coeffs_t, wavelet)
    x_rec = x_rec.astype(np.float32)
    return x_rec[:len(x)]

def dwt_denoise_multifeature(
    X, wavelet='db4', level=3, thresh_scale=0.7,
    soft_or_hard='soft', mode='smooth', eps=1e-8
):
    """
    Perform multifeature denoising using DWT (causal-safe) with mode='smooth'.

    Parameters
    ----------
    X : np.ndarray
        (num_sequences, seq_len, num_features)
    wavelet : str
        Wavelet basis (e.g. 'db4', 'bior1.3', etc.)
    level : int
        Decomposition level
    thresh_scale : float
        Multiplier for universal threshold
    soft_or_hard : str
        Thresholding mode ('soft' or 'hard')
    mode : str
        Edge mode passed to pywt.wavedec (default 'smooth')
    eps : float
        Small constant for numerical safety / clipping

    Returns
    -------
    denoised : np.ndarray
        Same shape as X, length preserved.
    """
    num_seq, seq_len, num_feat = X.shape
    # Preallocate with the final dtype
    out = np.empty_like(X, dtype=np.float32)

    for f in range(num_feat):
        for s in range(num_seq):
            x = X[s, :, f].astype(float)

            coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=mode)

            d1 = coeffs[-1]
            sigma = np.median(np.abs(d1)) / 0.6745 + eps
            lam = thresh_scale * sigma * np.sqrt(2 * np.log(len(x)))

            coeffs_thr = [coeffs[0]] + [
                pywt.threshold(c, lam, mode=soft_or_hard) for c in coeffs[1:]
            ]

            x_rec = pywt.waverec(coeffs_thr, wavelet=wavelet, mode=mode)

            if len(x_rec) > len(x):
                x_rec = x_rec[:len(x)]
            elif len(x_rec) < len(x):
                x_rec = np.pad(x_rec, (0, len(x) - len(x_rec)), mode='edge')

            # Clip to keep strictly positive for downstream logs
            x_rec = np.clip(x_rec, eps, None)

            # Write directly as float32
            out[s, :, f] = x_rec.astype(np.float32)

    return out

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

"""
 - DO NOT denoise volume or num trades - spikes are signal!!
 - DO NOT denoise technical indicators - they are already pretty smooth
"""

#### Initial Data Processing ####
def load_data(fname):
    df = pd.read_csv(fname)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    if SELECT_RANGE:
        START_DATE = pd.Timestamp('2022-01-01')  # change if you want a different cutoff
        df = (
            df[df['close_time'] >= START_DATE]
            .sort_values('close_time')
            .reset_index(drop=True)
        )

    print(f"Loaded {len(df)} rows from CSV")
    print(f"Date range: {df['close_time'].min().strftime('%m/%d/%Y')} to {df['close_time'].max().strftime('%m/%d/%Y')}")

    return df

def process_columns(df):
    # Bollinger Band Width and Position
    df['BB_width'] =  (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0']) / (df['BBM_20_2.0_2.0'] + 1e-8)
    df['BB_position'] = (df['close'] - df['BBL_20_2.0_2.0']) / (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0'] + 1e-8)

    # MACD
    if USE_MACD:
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        df.drop(columns=['MACD_12_26_9', 'MACDs_12_26_9'], inplace=True)

    # Standard Moving Average
    df["SMA_14"] = ta.sma(df["close"], length=14)
    df = df.dropna(axis=0, how='any').reset_index(drop=True)

    # Apply denoising to OHLC data if enabled
    if DENOISE:
        OHLC = df.iloc[:, :4].to_numpy()
        original_close = df.iloc[:, 3].copy()  # Save original for comparison
        denoised_OHLC = kalman_denoise_multifeature(OHLC)

        n_original = len(df)
        n_denoised = len(denoised_OHLC)
        if n_denoised < n_original:
            trim = n_original - n_denoised
            df = df.iloc[trim:].reset_index(drop=True)
            original_close = original_close.iloc[trim:].reset_index(drop=True)
            denoised_OHLC = denoised_OHLC[-len(df):]

        # Replace the first 4 columns with the denoised values
        df.iloc[:, :4] = denoised_OHLC

        # Visualize denoising effect on close price (random 1000-point window)
        if len(df) > 1000:
            start_idx = np.random.randint(0, len(df) - 1000)
            window_slice = slice(start_idx, start_idx + 1000)
        else:
            window_slice = slice(0, len(df))

        plt.figure(figsize=(14, 6))
        plt.plot(original_close.iloc[window_slice].values, label='Original Close Price', linewidth=1, alpha=0.6, color='gray')
        plt.plot(df.iloc[window_slice, 3].values, label='Denoised Close Price (Kalman)', linewidth=1.5, alpha=0.9, color='blue')
        plt.title(f'Denoising Effect on Close Price (Random {len(df.iloc[window_slice])} datapoints, BEFORE log returns)\nQ/R ratio: {1e-5 / 1e-3:.4f}', fontsize=14)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('denoised_close_price_preview.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved denoising comparison to: denoised_close_price_preview.png")
        print(f"   Window: samples {window_slice.start} to {window_slice.stop}")
        print(f"   Kalman Q/R ratio: {1e-5 / 1e-3:.4f} (very heavy smoothing)\n")
        plt.show()

    # Check for negative values in the first 4 columns
    neg_mask = (df.iloc[:, :4] < 0)
    print(f"Number of negative values: {neg_mask.sum().sum()}")

    # Convert to log returns (first 4 columns only)
    # All OHLC returns computed relative to previous close price
    EPS = 1e-8
    prev_close = df.iloc[:, 3].shift(1)  # Previous close (4th column, index 3)
    log_returns = np.log(np.clip(df.iloc[:, :4], EPS, None).div(np.clip(prev_close, EPS, None), axis=0))
    log_returns = log_returns.fillna(0)
    df.iloc[:, :4] = log_returns
    df.columns = ['log_return_open', 'log_return_high', 'log_return_low', 'log_return_close'] + list(df.columns[4:])

    ### REMOVE UN-NEEDED COLUMNS ###
    close_times = df['close_time']
    close_times.to_csv('datapoint_timestamps.csv', index=True, header=True)
    df.drop(columns=['close_time', 'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0', 'number_of_trades'], inplace=True)

    if USE_MACD:
        df = df.iloc[33:, :]  # ONLY use FOR MACD

    df.columns.to_series().to_csv("column_names.csv", index=False, header=False)
    arr = df.to_numpy()
    close_idx = df.columns.get_loc('log_return_close')
    print(f"Columns After Processing: {df.columns.tolist()}\n\n")

    return df.columns.tolist(), arr, close_idx


#### Functions for Shuffling ####
def block(arr, BLOCK_SIZE):
    """
    Accepts: (num_datapoints, num_features)-size numpy array
    Returns: (num_blocks, block_size, num_features)-size numpy array

    Where num_blocks = floor(num_datapoints/block_size), and the remaining
    earlier datapoints are dropped so that each block is the same size
    """
    num_datapoints, num_features = arr.shape
    num_blocks = num_datapoints // BLOCK_SIZE  # floor division

    # Drop the earlier datapoints so that we can evenly reshape
    trimmed_arr = arr[-num_blocks * BLOCK_SIZE:, :]

    # Reshape into blocks
    blocked_data = trimmed_arr.reshape(num_blocks, BLOCK_SIZE, num_features)

    return blocked_data

def shuffle_blocks(X, Y):
    perm = rng.permutation(len(X))  # ensures both get shuffled in the same way
    return X[perm], Y[perm]

def block_and_sequence(arr, BLOCK_SIZE, close_idx):
    blocked_data = block(arr, BLOCK_SIZE=BLOCK_SIZE)
    num_blocks, block_size, num_features = blocked_data.shape
    num_sequences = block_size - ENCODER_LENGTH_P - HORIZON_T
    if num_sequences <= 0:
        raise ValueError(f"BLOCK_SIZE ({BLOCK_SIZE}) must be > ENCODER_LENGTH_P+HORIZON_T ({ENCODER_LENGTH_P+HORIZON_T})")

    blocked_Xs = np.zeros((num_blocks, num_sequences, ENCODER_LENGTH_P, num_features))
    blocked_Ys = np.zeros((num_blocks, num_sequences, HORIZON_T, num_features))

    for i in range(num_blocks):
        block_i = blocked_data[i, :, :]
        block_X, block_Y = sequence(block_i, ENCODER_LENGTH_P, HORIZON_T, close_idx)
        blocked_Xs[i] = block_X
        blocked_Ys[i] = block_Y

    return blocked_Xs, blocked_Ys


#### Critical Steps: Sequence, Split, Normalize ####
def sequence(data, window_P, horizon_T, close_idx):
    """
    Create sequences for encoder-decoder training.

    Parameters
    ----------
    data : np.ndarray
        Shape (T_total, D) - time series data
    window_P : int
        Encoder length (past context)
    horizon_T : int
        Decoder length (future to predict)
    close_idx : int
        Index of close price feature

    Returns
    -------
    X : np.ndarray
        Shape (num_sequences, window_P, D) - encoder inputs
    Y : np.ndarray
        Shape (num_sequences, horizon_T, D) - decoder targets (full feature vectors)
    """
    T = data.shape[0]
    D = data.shape[1]
    starts = np.arange(0, T - window_P - horizon_T + 1)

    # Build encoder sequences (X) - past P timesteps
    X = np.stack([data[i:i+window_P] for i in starts], axis=0)

    # Build decoder sequences (Y) - future T timesteps with ALL features
    Y = np.stack([data[i+window_P:i+window_P+horizon_T] for i in starts], axis=0)

    return X, Y

def split(array, train_frac, val_frac):
    train_size = int(len(array) * train_frac)
    val_size = int(len(array) * val_frac)

    train_set = array[:train_size]
    val_set = array[train_size:train_size + val_size]
    test_set = array[train_size + val_size:]

    return train_set, val_set, test_set

def normalize_wrt_train(train, val, test, eps=1e-8):
    """
    Normalize datasets (train, val, test) using mean/std from training set only.
    Works for arrays of shape:
      - (N, F)
      - (N, T, F)
      - (B, S, T, F)
    Normalization is applied feature-wise across *all* sequences and timesteps.

    Returns:
        train_norm, val_norm, test_norm, mu, sigma
    """
    # Determine which axes to average over (all but last)
    axes = tuple(range(train.ndim - 1))

    # Compute feature-wise mean/std using train set only
    mu = np.mean(train, axis=axes, keepdims=True)
    sigma = np.std(train, axis=axes, keepdims=True) + eps

    # Apply normalization
    train_norm = (train - mu) / sigma
    val_norm   = (val   - mu) / sigma
    test_norm  = (test  - mu) / sigma

    return train_norm, val_norm, test_norm, mu.squeeze(), sigma.squeeze()

def verify_normalization(train_set, val_set, test_set):
    """
    Prints feature-wise means and stds for any-shaped datasets.
    Works for arrays of shape:
        (N, F), (N, T, F), or (B, S, T, F)
    """
    print("Number of features:", train_set.shape[-1])

    def summarize(name, arr):
        # Average over all axes except the last (features)
        axes = tuple(range(arr.ndim - 1))
        mean = np.mean(arr, axis=axes)
        std = np.std(arr, axis=axes)
        print(f"{name} means: {[f'{x:.3f}' for x in mean]}")
        print(f"{name} stds:  {[f'{x:.3f}' for x in std]}")

    summarize("Train", train_set)
    summarize("Val", val_set)
    summarize("Test", test_set)
    print("\n")


#### Last step, Batching ####
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

    # We must store the last batch separately, because it may be smaller than batch_size
    Last_X_batch = X_batches[-1]
    Last_Y_batch = Y_batches[-1]
    X_batches.pop()
    Y_batches.pop()

    return np.array(X_batches), np.array(Y_batches), np.array(Last_X_batch), np.array(Last_Y_batch)

def summarize_batched_data(X_train_batches, Y_train_batches,
                           X_val_batches,   Y_val_batches,
                           X_test_batches,  Y_test_batches):
    """
    Summarizes batched datasets (train/val/test) by printing key info.
    """

    # Infer common dimensions from the first batch
    batch_size, seq_len_P, num_features = X_train_batches[0].shape
    _, seq_len_T, _ = Y_train_batches[0].shape

    num_batches_train = len(X_train_batches)
    num_batches_val   = len(X_val_batches)
    num_batches_test  = len(X_test_batches)

    # Total examples (no last batch assumption)
    n_train = num_batches_train * batch_size
    n_val   = num_batches_val   * batch_size
    n_test  = num_batches_test  * batch_size
    total   = n_train + n_val + n_test

    # Fractions
    f_train = n_train / total
    f_val   = n_val   / total
    f_test  = n_test  / total

    # ---- Pretty printing ----
    print("\n" + "#"*65)
    print(f"{'DATA SUMMARY':^65}")
    print("#"*65 + "\n")

    print(f"Number of features: {num_features}")
    print(f"Encoder sequence length (P): {seq_len_P}")
    print(f"Decoder sequence length (T): {seq_len_T}")
    print(f"Batch size: {batch_size}")
    print()

    print("Dataset Split Summary:")
    print("-" * 65)
    print(f"{'Set':<10}{'Num Batches':<15}{'Examples':<15}{'Fraction':<15}")
    print("-" * 65)
    print(f"{'Train':<10}{num_batches_train:<15}{n_train:<15}{f_train:<15.3f}")
    print(f"{'Val':<10}{num_batches_val:<15}{n_val:<15}{f_val:<15.3f}")
    print(f"{'Test':<10}{num_batches_test:<15}{n_test:<15}{f_test:<15.3f}")
    print("-" * 65)

    print("\nExample batch shapes:")
    print(f"X_train_batches[0]: {X_train_batches[0].shape}, Y_train_batches[0]: {Y_train_batches[0].shape}")
    print(f"X_val_batches[0]:   {X_val_batches[0].shape}, Y_val_batches[0]:   {Y_val_batches[0].shape}")
    print(f"X_test_batches[0]:  {X_test_batches[0].shape}, Y_test_batches[0]:  {Y_test_batches[0].shape}")

    print("\n" + "#"*65 + "\n")


if __name__ == '__main__':
    # ============================================================
    # 1) Load data from CSV
    # ============================================================
    df = load_data('../Data Processing/BTCUSDT_30m_10years.csv')

    if COMBINE_DATA:
        df2 = load_data('../Data Processing/ETHUSDT_30m_10years.csv')

    # ============================================================
    # 2) Add New Technical Indicators and compute log returns
    # ============================================================
    col_names, arr, close_idx = process_columns(df)
    NUM_FEATURES = arr.shape[1]

    if COMBINE_DATA:
        col_names2, arr2, _ = process_columns(df2)

    print("\n#####################################################\n")

    """
    If we shuffle blocks, we do:
    - BLOCK --> SEQUENCE BLOCKS --> SHUFFLE --> SPLIT --> NORM --> BATCH
    """
    if BLOCK_SHUFFLE:

        blocked_Xs, blocked_Ys = block_and_sequence(arr, BLOCK_SIZE, close_idx)

        if COMBINE_DATA:
            blocked_Xs2, blocked_Ys2 = block_and_sequence(arr2, BLOCK_SIZE, close_idx)
            combined_block_Xs = np.concatenate((blocked_Xs, blocked_Xs2), axis=0)
            combined_block_Ys = np.concatenate((blocked_Ys, blocked_Ys2), axis=0)
            shuffled_combined_blocked_Xs, shuffled_combined_blocked_Ys = shuffle_blocks(combined_block_Xs, combined_block_Ys)

            X = shuffled_combined_blocked_Xs
            Y = shuffled_combined_blocked_Ys

        else:
            shuffled_blocked_Xs, shuffled_blocked_Ys = shuffle_blocks(blocked_Xs, blocked_Ys)
            X = shuffled_blocked_Xs
            Y = shuffled_blocked_Ys

        # Each is (num_blocks, sequences_per_block, sequence_length, num_features)
        X_train, X_val, X_test = split(X, train_frac, val_frac)
        Y_train, Y_val, Y_test = split(Y, train_frac, val_frac)

        # Combine all blocks - they are no longer needed
        # Becomes (num_sequences, sequence_length, num_features)
        X_train = X_train.reshape(-1, X_train.shape[2], X_train.shape[3])
        Y_train = Y_train.reshape(-1, Y_train.shape[2], Y_train.shape[3])
        X_val   = X_val.reshape(-1, X_val.shape[2], X_val.shape[3])
        Y_val   = Y_val.reshape(-1, Y_val.shape[2], Y_val.shape[3])
        X_test  = X_test.reshape(-1, X_test.shape[2], X_test.shape[3])
        Y_test  = Y_test.reshape(-1, Y_test.shape[2], Y_test.shape[3])

        X_train, X_val, X_test, mu, sigma = normalize_wrt_train(X_train, X_val, X_test)
        Y_train, Y_val, Y_test, _, _ = normalize_wrt_train(Y_train, Y_val, Y_test)

        print(X_train.shape, X_val.shape, X_test.shape)


    # No shuffling
    else:
        # ============================================================
        # 4) Split, Normalize, and Sequence
        # ============================================================
        """
        If no shuffling, we do:
        - SPLIT --> SEQUENCE --> NORM --> BATCH
        """

        train, val, test = split(arr, train_frac, val_frac)

        if COMBINE_DATA:
            train2, val2, test2 = split(arr2, train_frac, val_frac)
            train = np.concatenate((train, train2), axis=0)
            val = np.concatenate((val, val2), axis=0)
            test = np.concatenate((test, test2), axis=0)

        print("Starting Sequencing...")
        X_train, Y_train = sequence(train, ENCODER_LENGTH_P, HORIZON_T, close_idx)
        X_val, Y_val =     sequence(val, ENCODER_LENGTH_P, HORIZON_T, close_idx)
        X_test, Y_test =   sequence(test, ENCODER_LENGTH_P, HORIZON_T, close_idx)

        print(f"Sequence shapes before normalization:")
        print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
        print(f"X_val: {X_val.shape}, Y_val: {Y_val.shape}")
        print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}\n")

        # Normalize X and Y separately but using same statistics from training set
        X_train, X_val, X_test, mu_x, sigma_x = normalize_wrt_train(X_train, X_val, X_test)
        Y_train, Y_val, Y_test, mu_y, sigma_y = normalize_wrt_train(Y_train, Y_val, Y_test)

        verify_normalization(X_train, X_val, X_test)

        # Save normalization statistics
        mu_sigma_df = pd.DataFrame({
            'mu': mu_x,
            'sigma': sigma_x
        })
        mu_sigma_df.to_csv('mu_sigma_df.csv', index=False)
        print("Saved normalization statistics to mu_sigma_df.csv")

    # ============================================================
    # 7) Create Batches of size `batch_size`
    # ============================================================
    print("Starting batching...")
    X_train_batches, Y_train_batches, _, _ = create_batches(X_train, Y_train, batch_size)
    X_val_batches, Y_val_batches, _, _ = create_batches(X_val, Y_val, batch_size)
    X_test_batches, Y_test_batches, _, _ = create_batches(X_test, Y_test, batch_size)
    print("Batching complete")

    summarize_batched_data(X_train_batches, Y_train_batches,
                            X_val_batches,   Y_val_batches,
                            X_test_batches,  Y_test_batches)

    # ============================================================
    # 8) Save Data
    # ============================================================
    fname = f"preprocessed_data{'_COMBINED' if COMBINE_DATA else ''}{'_SHUFFLED' if BLOCK_SHUFFLE else ''}.npz"

    np.savez_compressed(fname,
        X_train_batches=X_train_batches, Y_train_batches=Y_train_batches,
        X_val_batches=X_val_batches, Y_val_batches=Y_val_batches,
        X_test_batches=X_test_batches, Y_test_batches=Y_test_batches,
    )

    print(f"Saved {fname}")
