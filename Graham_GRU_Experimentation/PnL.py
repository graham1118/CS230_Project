# ===============================================================
#   PnL Evaluation and Simulation for EncoderGRU (auto-align Y)
# ===============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# Paths (match your repo layout)
# -----------------------------
DATA_PATH      = "../Data Processing/preprocessed_data.npz"
RAW_CSV        = "../Data Processing/BTCUSDT_30m_10years.csv"
TIMESTAMPS_CSV = "../Data Processing/datapoint_timestamps.csv"
MU_SIGMA_CSV   = "../Data Processing/mu_sigma_df.csv"
SAVE_PATH      = "cur_best_model.pth"

# -----------------------------
# Constants (match preprocessing)
# -----------------------------
HORIZON   = 2
SEQ_LEN   = 16
CLOSE_IDX = 3  # 'log_return_close' in your processed df

# -----------------------------
# Model hyperparams (as trained)
# -----------------------------
LATENT_SIZE      = 32
GRU_HIDDEN_SIZE  = 64
GRU_OUTPUT_SIZE  = 1
NUM_GRU_LAYERS   = 1
INIT_LEARNING_RATE = 1e-3

# -----------------------------
# PnL sim params
# -----------------------------
START_BALANCE = 10_000.0
SLIPPAGE_BP   = 0 #5
FEE_BP        = 0 #5
TRADE_EVERY   = HORIZON

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================
# Utilities
# ===============================================================
def log_returns_to_prices(log_rets, initial_price):
    """
    Convert de-normalized log returns to absolute prices.
    """
    log_rets = np.asarray(log_rets).flatten()
    cum_log = np.cumsum(log_rets)
    return float(initial_price) * np.exp(cum_log)


def flatten_batches(X_batches, Y_batches):
    """
    (nb, B, T, F) -> (N, T, F)
    (nb, B, 1)    -> (N, 1)
    """
    X_all = np.concatenate([b for b in X_batches], axis=0)
    Y_all = np.concatenate([b for b in Y_batches], axis=0)
    return X_all, Y_all


# ===============================================================
# Load model (EncoderGRU) exactly as in training
# ===============================================================
def load_checkpointed_gru(num_features, SAVE_PATH, INIT_LEARNING_RATE=1e-3,
                          GRU_HIDDEN_SIZE=64, GRU_OUTPUT_SIZE=1,
                          NUM_GRU_LAYERS=1, LATENT_SIZE=32):
    from GRU import EncoderGRU  # your class

    model = EncoderGRU(
        input_size=num_features,
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=GRU_OUTPUT_SIZE,
        num_layers=NUM_GRU_LAYERS,
        latent_size=LATENT_SIZE,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    checkpoint = torch.load(SAVE_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scheduler_state_dict", None) is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    val_loss = checkpoint.get("val_loss", float("nan"))
    print(f"✅ Loaded EncoderGRU from epoch {epoch}, val_loss={val_loss:.6f}")

    model.to(device)
    model.eval()
    return model


# ===============================================================
# Alignment check & auto-detection of Y inverse
# ===============================================================
def alignment_check_and_pick_inverse(
    y_true_norm_like, y_pred_norm_like, last_x_norm,
    mu_close, sigma_close, timestamps_csv, raw_csv,
    train_frac=0.9, val_frac=0.05
):
    """
    We don't rely on assumptions about whether Y was normalized.
    We try BOTH inverses:

      A) Y was normalized & 1000-scaled:
           invA(z) = ((z) * sigma_close + mu_close) / 1000
      B) Y was ONLY 1000-scaled (no normalization):
           invB(z) = (z) / 1000

    We apply the same inverse to both true and predicted series (after forming
    the predicted full target = pred_resid + last_x_norm), reconstruct prices,
    and choose the inverse that produces higher correlation with RAW price path.
    """
    # --- timestamps for Y indices (test portion) ---
    ts_df = pd.read_csv(timestamps_csv, index_col=0)
    time_col = [c for c in ts_df.columns if "time" in c.lower()][0]
    ts = pd.to_datetime(ts_df[time_col])

    n_total = len(ts)
    train_end = int(train_frac * n_total)
    val_end = int((train_frac + val_frac) * n_total)
    test_start = val_end

    # Y refers to indices starting at SEQ_LEN + HORIZON after the split start
    y_idx_start = test_start + SEQ_LEN + HORIZON
    y_times = ts.iloc[y_idx_start : y_idx_start + len(y_true_norm_like)]

    # Load raw closes aligned to y_times
    raw = pd.read_csv(raw_csv)
    raw["close_time"] = pd.to_datetime(raw["close_time"], unit="ms")
    joined = pd.merge(
        pd.DataFrame({"close_time": y_times.values}),
        raw[["close_time", "close"]],
        on="close_time", how="left"
    ).ffill().bfill()
    actual_prices = joined["close"].to_numpy(dtype=float)

    # Helper: evaluate one inverse
    def evaluate_inverse(inv_name, inv_fn):
        # Undo for TRUE
        y_true_logret = inv_fn(y_true_norm_like)
        # Undo for PRED: first form predicted *normalized* target as in training:
        # pred_full_norm_like = pred_resid_norm_like + last_x_norm
        y_pred_full_norm_like = y_pred_norm_like + last_x_norm
        y_pred_logret = inv_fn(y_pred_full_norm_like)

        # Reconstruct price series from log-returns
        recon_true = log_returns_to_prices(y_true_logret, initial_price=actual_prices[0])

        # Correlation as fit score
        m = min(len(actual_prices), len(recon_true))
        corr = np.corrcoef(actual_prices[:m], recon_true[:m])[0, 1]

        return corr, y_true_logret, y_pred_logret, recon_true

    # Define both candidate inverses
    invA = lambda z: ((z) * sigma_close + mu_close) / 1000.0
    invB = lambda z: (z) / 1000.0

    corrA, y_true_A, y_pred_A, recon_A = evaluate_inverse("A", invA)
    corrB, y_true_B, y_pred_B, recon_B = evaluate_inverse("B", invB)

    print(f"[Auto-Detect] Corr with RAW using inverse A (norm+scale): {corrA:.6f}")
    print(f"[Auto-Detect] Corr with RAW using inverse B (scale only): {corrB:.6f}")

    # Choose the better inverse
    if np.nan_to_num(corrA, nan=-1) >= np.nan_to_num(corrB, nan=-1):
        chosen = "A (normalized + 1000×)"
        y_true_logret, y_pred_logret, recon_prices = y_true_A, y_pred_A, recon_A
    else:
        chosen = "B (1000× only)"
        y_true_logret, y_pred_logret, recon_prices = y_true_B, y_pred_B, recon_B

    print(f"👉 Chosen inverse: {chosen}")

    # Print quick head for verification
    print("\n=== Alignment Verification (first 10 samples) ===")
    chk = pd.DataFrame({
        "timestamp": y_times.iloc[:10].dt.strftime("%Y-%m-%d %H:%M"),
        "raw_close": actual_prices[:10],
        "reconstructed": recon_prices[:10]
    })
    print(chk)

    plt.figure(figsize=(11,4))
    plt.plot(actual_prices[-300:], label="Raw Close (CSV)")
    plt.plot(recon_prices[-300:], "--", label="Reconstructed from Y")
    plt.title("Alignment Check: Raw vs Reconstructed (Test Y)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return actual_prices, y_times.to_numpy(), y_true_logret, y_pred_logret


# ===============================================================
# PnL simulation
# ===============================================================
def pnl_simulation_from_returns(
    actual_prices, y_true_logret, y_pred_logret,
    trade_stride=HORIZON, start_balance=START_BALANCE,
    slippage_bp=SLIPPAGE_BP, fee_bp=FEE_BP
):
    slip = slippage_bp / 1e4
    fee  = fee_bp / 1e4

    balance = float(start_balance)
    equity_curve = []
    trades = []

    N = len(y_true_logret)
    for i in range(0, N - trade_stride, trade_stride):
        p_now = actual_prices[i]
        p_future = actual_prices[i + trade_stride]

        # Direction by predicted horizon log-return sign (use last step within stride)
        pred_ret = float(y_pred_logret[i + trade_stride - 1])
        direction = 1 if pred_ret > 0.0 else -1

        actual_ret = (p_future / p_now - 1.0) * direction
        net_ret = actual_ret - (slip + fee)

        balance *= (1.0 + net_ret)
        equity_curve.append(balance)
        trades.append({
            "i": i,
            "dir": direction,
            "p_now": p_now,
            "p_future": p_future,
            "gross_ret": actual_ret,
            "net_ret": net_ret,
            "balance": balance
        })

    return pd.DataFrame(trades), np.array(equity_curve, dtype=float)


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    # Load preprocessed (batched) data
    data = np.load(DATA_PATH)
    X_test_batches = data["X_test_batches"]   # (nb, B, T, F)
    Y_test_batches = data["Y_test_batches"]   # (nb, B, 1)
    NUM_FEATURES = X_test_batches.shape[-1]
    print("Loaded test data:", X_test_batches.shape, Y_test_batches.shape)

    # Load mu/sigma (feature stats) — used if Y was normalized like X
    mu_sigma_df = pd.read_csv(MU_SIGMA_CSV)
    mu = mu_sigma_df["mu"].to_numpy()
    sigma = mu_sigma_df["sigma"].to_numpy()
    mu_close, sigma_close = mu[CLOSE_IDX], sigma[CLOSE_IDX]
    print(f"mu_close={mu_close:.6e}, sigma_close={sigma_close:.6e}")

    # Build model & load checkpoint
    model = load_checkpointed_gru(
        num_features=NUM_FEATURES,
        SAVE_PATH=SAVE_PATH,
        INIT_LEARNING_RATE=INIT_LEARNING_RATE,
        GRU_HIDDEN_SIZE=GRU_HIDDEN_SIZE,
        GRU_OUTPUT_SIZE=GRU_OUTPUT_SIZE,
        NUM_GRU_LAYERS=NUM_GRU_LAYERS,
        LATENT_SIZE=LATENT_SIZE,
    )

    # Flatten test to (N, T, F) and (N, 1)
    X_test_all, Y_test_all = flatten_batches(X_test_batches, Y_test_batches)
    print("Flattened:", X_test_all.shape, Y_test_all.shape)

    # Forward pass — NOTE: model trained to predict residuals, so we must add last step later
    with torch.no_grad():
        X_t = torch.tensor(X_test_all, dtype=torch.float32, device=device)
        pred_resid = model(X_t).cpu().numpy().squeeze(-1)           # (N,)
        last_x_norm = X_test_all[:, -1, CLOSE_IDX]                  # last *normalized-like* close feature
    y_true_norm_like = Y_test_all.squeeze(-1)                       # stored target (same normalization regime used in training)

    # Auto-detect correct inverse (normalized+scaled vs scaled only) and align
    actual_prices, y_times, y_true_logret, y_pred_logret = alignment_check_and_pick_inverse(
        y_true_norm_like=y_true_norm_like,
        y_pred_norm_like=pred_resid,
        last_x_norm=last_x_norm,
        mu_close=mu_close,
        sigma_close=sigma_close,
        timestamps_csv=TIMESTAMPS_CSV,
        raw_csv=RAW_CSV
    )

    # PnL simulation
    trades_df, equity_curve = pnl_simulation_from_returns(
        actual_prices=actual_prices,
        y_true_logret=y_true_logret,
        y_pred_logret=y_pred_logret,
        trade_stride=TRADE_EVERY,
        start_balance=START_BALANCE,
        slippage_bp=SLIPPAGE_BP,
        fee_bp=FEE_BP
    )

    # Plot equity
    plt.figure(figsize=(10,4))
    plt.plot(equity_curve, label="Equity")
    plt.title(f"PnL Simulation (stride={TRADE_EVERY}, slp={SLIPPAGE_BP}bp, fee={FEE_BP}bp)")
    plt.xlabel(f"Trades (every {TRADE_EVERY} × 30 min)")
    plt.ylabel("USD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Summary
    total_return = equity_curve[-1] / START_BALANCE - 1.0 if len(equity_curve) else 0.0
    step_rets = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0.0])
    sharpe_like = (np.mean(step_rets) / (np.std(step_rets) + 1e-12)) if step_rets.size else 0.0

    print("\n========== PnL SUMMARY ==========")
    print(f"Final Balance: ${equity_curve[-1]:.2f}" if len(equity_curve) else f"Final Balance: ${START_BALANCE:.2f}")
    print(f"Total Return: {100*total_return:.2f}%")
    print(f"Sharpe-like: {sharpe_like:.3f}")
    print(f"Num trades:  {len(trades_df)}")
    if not trades_df.empty:
        print(trades_df.head())

    # Save logs
    trades_df.to_csv("trades_log.csv", index=False)
    pd.DataFrame({"equity": equity_curve}).to_csv("equity_curve.csv", index=False)
    print("✅ Saved trades_log.csv and equity_curve.csv")
