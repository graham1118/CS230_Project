
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
from matplotlib import pyplot as plt
import os
from typing import Tuple

# ----------------------------
# Device & reproducibility
# ----------------------------
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("USING DEVICE:", device)

def save_object_to_file(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def read_object_from_file(filepath):
    with open(filepath, "rb") as f:
        out = pickle.load(f)
    return out

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





# ============================================================
# Configuration
# ============================================================
SEED = 0
EPOCHS = 20
results_save_path = "exp_results.pkl"
plot_save_path_suffix = "_loss_plot.jpg"

NPZ_PATH = "preprocessed_data.npz"  # produced by your preprocessing script

# What to predict:
#   "return" -> future log return: log( C_{t+h} / C_t )
#   "close"  -> raw future close (discouraged; non-stationary)
TARGET_MODE = "return"  # <<< RECOMMENDED

# Column index of 'close' in your rows (based on your preprocessing ordering)
CLOSE_COL_INDEX = 3

# Optimization
INIT_LR = 3e-4          # smaller LR helps stability
WEIGHT_DECAY = 0
BATCH_SIZE = 256
DROPOUT = 0.10
CLIP_GRAD_NORM = 1.0    # gradient clipping to prevent explosion
USE_SCHEDULER = True    # ReduceLROnPlateau on val loss

# End-of-run evaluation options
RUN_PNL = True
TAKER_FEE_BPS = 5.0      # 5 bp per trade per side (set 0 for fee-less toy PnL)
PROBA_THRESHOLD = 0.0    # sign threshold; keep 0 for simple sign(pred)


# ============================================
# Load & reshape preprocessed batches from NPZ
# ============================================
def _merge_X_batches(X_batches, X_last):
    if X_batches.size == 0:
        merged_full = np.empty((0, *X_last.shape[1:]), dtype=X_last.dtype)
    else:
        merged_full = X_batches.reshape(-1, X_batches.shape[-2], X_batches.shape[-1])
    return np.concatenate([merged_full, X_last], axis=0)

def _merge_Y_batches(Y_batches, Y_last):
    if Y_batches.size == 0:
        merged_full = np.empty((0, Y_last.shape[-1]), dtype=Y_last.dtype)
    else:
        merged_full = Y_batches.reshape(-1, Y_batches.shape[-1])
    return np.concatenate([merged_full, Y_last], axis=0)

def load_sequences_and_targets(npz_path: str, target_mode: str, close_col: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found at {os.path.abspath(npz_path)}")

    data = np.load(npz_path, allow_pickle=True)

    # sequences and future rows
    X_train_seq = _merge_X_batches(data["X_train_batches"], data["Last_X_train_batch"])
    X_val_seq   = _merge_X_batches(data["X_val_batches"],   data["Last_X_val_batch"])
    X_test_seq  = _merge_X_batches(data["X_test_batches"],  data["Last_X_test_batch"])

    Y_train_full = _merge_Y_batches(data["Y_train_batches"], data["Last_Y_train_batch"])
    Y_val_full   = _merge_Y_batches(data["Y_val_batches"],   data["Last_Y_val_batch"])
    Y_test_full  = _merge_Y_batches(data["Y_test_batches"],  data["Last_Y_test_batch"])

    W = X_train_seq.shape[1]
    Fdim = X_train_seq.shape[2]
    input_dim = W * Fdim

    # Build scalar targets
    if target_mode == "close":
        y_train = Y_train_full[:, close_col:close_col+1].astype(np.float32)
        y_val   = Y_val_full[:,   close_col:close_col+1].astype(np.float32)
        y_test  = Y_test_full[:,  close_col:close_col+1].astype(np.float32)
    elif target_mode == "return":
        eps = 1e-12
        last_close_train = X_train_seq[:, -1, close_col]
        last_close_val   = X_val_seq[:,   -1, close_col]
        last_close_test  = X_test_seq[:,  -1, close_col]

        fut_close_train  = Y_train_full[:, close_col]
        fut_close_val    = Y_val_full[:,   close_col]
        fut_close_test   = Y_test_full[:,  close_col]

        y_train = (np.log(np.clip(fut_close_train, eps, None)) -
                   np.log(np.clip(last_close_train, eps, None))).astype(np.float32)[:, None]
        y_val   = (np.log(np.clip(fut_close_val,   eps, None)) -
                   np.log(np.clip(last_close_val,   eps, None))).astype(np.float32)[:, None]
        y_test  = (np.log(np.clip(fut_close_test,  eps, None)) -
                   np.log(np.clip(last_close_test,  eps, None))).astype(np.float32)[:, None]
    else:
        raise ValueError("TARGET_MODE must be 'return' or 'close'.")

    print(f"[INFO] Input dims: W={W}, F={Fdim}, flattened input_dim={input_dim}")
    print(f"[INFO] Target mode: {target_mode} (scalar), target_dim=1")
    print("Train seq shapes:", X_train_seq.shape, y_train.shape)
    print("Val   seq shapes:", X_val_seq.shape,   y_val.shape)
    print("Test  seq shapes:", X_test_seq.shape,  y_test.shape)

    return X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test, input_dim, 1


# ==================================
# Simple MLP for regression (scalar)
# ==================================
class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_f = layer_sizes[i]
            out_f = layer_sizes[i+1]
            layers.append(nn.Linear(in_f, out_f))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (batch, 1)

def loss_func(pred, target):
    return F.mse_loss(pred, target)


# ==========================
# Training / evaluation loop
# ==========================
def batch_iter(X, y, bs):
    for s in range(0, len(X), bs):
        yield X[s:s+bs], y[s:s+bs]

@torch.no_grad()
def eval_set(model, Xs, ys, batch_size):
    """Return (loss, mae, rmse, sign_acc)."""
    model.eval()
    n = len(Xs)
    loss_sum = 0.0
    abs_sum  = 0.0
    sq_sum   = 0.0
    correct  = 0
    total    = 0
    for xb, yb in batch_iter(Xs, ys, batch_size):
        pr = model(xb)
        loss_sum += float(loss_func(pr, yb)) * xb.size(0)
        abs_sum  += float((pr - yb).abs().mean()) * xb.size(0)
        sq_sum   += float(((pr - yb) ** 2).mean()) * xb.size(0)

        # directional accuracy (ignore exact zeros)
        pred_sign = torch.sign(pr)
        true_sign = torch.sign(yb)
        mask = (true_sign != 0)
        correct += int((pred_sign[mask] == true_sign[mask]).sum().item())
        total   += int(mask.sum().item())

    loss = loss_sum / max(1, n)
    mae  = abs_sum  / max(1, n)
    rmse = (sq_sum / max(1, n)) ** 0.5
    sign_acc = (correct / total) if total > 0 else float('nan')
    return loss, mae, rmse, sign_acc

def run_full_train(
    layer_sizes,
    dropout,
    learning_rate,
    weight_decay,
    batch_size,
    train_amt,
    epochs,
    X_train, y_train, X_val, y_val, X_test, y_test,
    verbose_freq = 1
):
    model = MLP(layer_sizes, dropout).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    ) if USE_SCHEDULER else None

    print("Currently running Model with architecture:")
    print(model)
    print("Using batch size:", batch_size, "| Total Epochs:", epochs)

    train_loss_hist, val_loss_hist, test_loss_hist = [], [], []
    best_val = float("inf")
    best_state = None

    # choose train subset if requested
    X_train_use = X_train[:train_amt]
    y_train_use = y_train[:train_amt]

    for epoch in range(epochs):
        # Shuffle once per epoch
        perm = torch.randperm(X_train_use.size(0), device=device)
        X_train_epoch = X_train_use[perm]
        y_train_epoch = y_train_use[perm]

        # -------- Train --------
        model.train()
        epoch_train_loss = 0.0
        total_grad_norm = 0.0
        nbatches = 0

        for xb, yb in batch_iter(X_train_epoch, y_train_epoch, batch_size):
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_func(preds, yb)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            total_grad_norm += float(grad_norm)
            nbatches += 1

            optimizer.step()
            epoch_train_loss += float(loss) * xb.size(0)

        epoch_train_loss /= len(X_train_epoch)
        avg_grad_norm = total_grad_norm / max(1, nbatches)
        train_loss_hist.append(epoch_train_loss)

        # -------- Evaluate (Val & Test) --------
        val_loss, val_mae, val_rmse, val_sign = eval_set(model, X_val, y_val, batch_size)
        test_loss, test_mae, test_rmse, test_sign = eval_set(model, X_test, y_test, batch_size)

        val_loss_hist.append(val_loss)
        test_loss_hist.append(test_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose_freq is not None and epoch % verbose_freq == 0:
            if TARGET_MODE == "return":
                mae_bp = val_mae * 1e4
                rmse_bp = val_rmse * 1e4
                sign_pct = (val_sign * 100.0) if not np.isnan(val_sign) else float('nan')
                print(f"Epoch {epoch:3d} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                      f"train {epoch_train_loss:.6g} | "
                      f"val {val_loss:.6g} (MAE {val_mae:.3g}, {mae_bp:.1f} bp; "
                      f"RMSE {val_rmse:.3g}, {rmse_bp:.1f} bp; "
                      f"DirAcc {sign_pct:.2f}%) | "
                      f"test {test_loss:.6g} | grad|| {avg_grad_norm:.3g}")
            else:
                print(f"Epoch {epoch:3d} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                      f"train {epoch_train_loss:.6g} | val {val_loss:.6g} "
                      f"(MAE {val_mae:.6g}, RMSE {val_rmse:.6g}, DirAcc {val_sign*100:.2f}%) | "
                      f"test {test_loss:.6g} | grad|| {avg_grad_norm:.3g}")

    # Load best-by-val-loss
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, (train_loss_hist, val_loss_hist, test_loss_hist)


# ------------- Final evaluation helpers -------------
@torch.no_grad()
def predict(model, X):
    model.eval()
    preds = []
    for s in range(0, len(X), 4096):
        preds.append(model(X[s:s+4096]).cpu().numpy())
    return np.vstack(preds).squeeze(-1)

def r2_score(y_true, y_pred):
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

def print_confusion_like(y_true, y_pred, thr=0.0):
    """Counts for sign agreement; ignores exact-zero truths."""
    ts = np.sign(y_true)
    ps = np.sign(y_pred - thr)
    mask = ts != 0
    ts = ts[mask]; ps = ps[mask]
    TP = np.sum((ts > 0) & (ps > 0))
    TN = np.sum((ts < 0) & (ps < 0))
    FP = np.sum((ts < 0) & (ps > 0))
    FN = np.sum((ts > 0) & (ps < 0))
    total = len(ts)
    acc = (TP + TN) / max(1, total)
    print(f"Direction Confusion-style (ignoring y==0):")
    print(f"  TP={TP} TN={TN} FP={FP} FN={FN} | DirAcc={acc*100:.2f}% (N={total})")

def tiny_pnl(y_true, y_pred, fee_bps=0.0, thr=0.0):
    """
    Simple long/short PnL:
      position = sign(pred - thr) in {-1,0,1}
      pnl_t = position * y_true - fee
    Fees: apply fee_bps on position change events (enter/flip/exit) as rough proxy.
    Returns cumulative log-PnL vector.
    """
    pos = np.sign(y_pred - thr)
    # Fee on position changes (enter/flip/exit)
    fee = 0.0
    if fee_bps > 0:
        changes = np.abs(np.diff(pos, prepend=0))  # 0->1, 1->-1, etc.
        # charge per change one “unit” fee in bps
        fee = (fee_bps / 1e4) * changes
    pnl = pos * y_true - fee
    return np.cumsum(pnl)

def save_plots_and_csv(tag, y_true, y_pred, pnl_curve=None):
    os.makedirs("artifacts", exist_ok=True)
    # 1) scatter
    plt.clf()
    plt.scatter(y_true, y_pred, s=2, alpha=0.3)
    plt.xlabel("True (log return)")
    plt.ylabel("Pred (log return)")
    plt.title(f"{tag} | Pred vs True")
    plt.grid(True, alpha=0.2)
    plt.savefig(f"artifacts/{tag}_scatter.png", dpi=160)

    # 2) errors histogram
    err = y_pred - y_true
    plt.clf()
    plt.hist(err, bins=80)
    plt.title(f"{tag} | Error histogram")
    plt.xlabel("Pred - True (log return)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.2)
    plt.savefig(f"artifacts/{tag}_err_hist.png", dpi=160)

    # 3) pnl
    if pnl_curve is not None:
        plt.clf()
        plt.plot(pnl_curve)
        plt.title(f"{tag} | Tiny long/short cumulative PnL (log units)")
        plt.xlabel("Samples")
        plt.ylabel("Cum PnL (log)")
        plt.grid(True, alpha=0.2)
        plt.savefig(f"artifacts/{tag}_pnl.png", dpi=160)

    # CSV
    np.savetxt(f"artifacts/{tag}_preds.csv",
               np.stack([y_true, y_pred], axis=1),
               delimiter=",", header="y_true_logret,y_pred_logret", comments="")


# ==========================
# Main: load, build, train
# ==========================
if __name__ == "__main__":
    seed_everything(SEED)
    print("[INFO] CWD:", os.getcwd())
    print("[INFO] Looking for NPZ at:", os.path.abspath(NPZ_PATH))

    # 1) Load sequences and build scalar targets (log-returns by default)
    Xtr_seq, Ytr, Xva_seq, Yva, Xte_seq, Yte, input_dim, target_dim = load_sequences_and_targets(
        NPZ_PATH, TARGET_MODE, CLOSE_COL_INDEX
    )

    # 2) Standardize FEATURES on TRAIN only; then flatten for MLP
    W = Xtr_seq.shape[1]
    Fdim = Xtr_seq.shape[2]

    def standardize_by_train(Xtr, Xva, Xte):
        tr_2d = Xtr.reshape(-1, Fdim)
        mu = tr_2d.mean(axis=0, keepdims=True)
        sigma = tr_2d.std(axis=0, keepdims=True) + 1e-6
        def apply(X):
            X2d = X.reshape(-1, Fdim)
            Xz = (X2d - mu) / sigma
            return Xz.reshape(X.shape).astype(np.float32)
        return apply(Xtr), apply(Xva), apply(Xte)

    Xtr_seq, Xva_seq, Xte_seq = standardize_by_train(Xtr_seq, Xva_seq, Xte_seq)

    X_train_np = Xtr_seq.reshape(len(Xtr_seq), W * Fdim).astype(np.float32)
    X_val_np   = Xva_seq.reshape(len(Xva_seq), W * Fdim).astype(np.float32)
    X_test_np  = Xte_seq.reshape(len(Xte_seq), W * Fdim).astype(np.float32)

    # 3) Torch tensors
    X_train_torch = torch.from_numpy(X_train_np).to(device)
    Y_train_torch = torch.from_numpy(Ytr.astype(np.float32)).to(device)
    X_val_torch   = torch.from_numpy(X_val_np).to(device)
    Y_val_torch   = torch.from_numpy(Yva.astype(np.float32)).to(device)
    X_test_torch  = torch.from_numpy(X_test_np).to(device)
    Y_test_torch  = torch.from_numpy(Yte.astype(np.float32)).to(device)

    print(f"[INFO] Final train shapes: X={X_train_torch.shape}, y={Y_train_torch.shape}")
    print(f"[INFO] Final val   shapes: X={X_val_torch.shape},   y={Y_val_torch.shape}")
    print(f"[INFO] Final test  shapes: X={X_test_torch.shape},  y={Y_test_torch.shape}")

    # 4) Define an MLP sized to your data
    exp_params = {
        "layer_sizes" : [input_dim, 512, 128, target_dim],
        "dropout"     : DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "learning_rate": INIT_LR,
        "batch_size"  : BATCH_SIZE,
        "train_amt"   : len(X_train_np),
        "epochs"      : EPOCHS
    }

    # 5) Train
    model, hists = run_full_train(
        exp_params["layer_sizes"], exp_params["dropout"], exp_params["learning_rate"], exp_params["weight_decay"],
        exp_params["batch_size"], exp_params["train_amt"], exp_params["epochs"],
        X_train_torch, Y_train_torch, X_val_torch, Y_val_torch, X_test_torch, Y_test_torch,
        verbose_freq=1
    )

    save_object_to_file({"mlp": hists}, results_save_path)
    # Plot curves
    train_hist, val_hist, test_hist = hists
    plt.clf()
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist,   label="Val")
    plt.plot(test_hist,  label="Test")
    plt.title(f"Loss vs Epochs (target={TARGET_MODE})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(loc="upper right")
    plt.savefig("learning_curves" + plot_save_path_suffix)
    plt.clf()

    # 6) Final evaluation on TEST (numbers you can compare across runs)
    y_true = Y_test_torch.cpu().numpy().squeeze(-1)
    y_pred = predict(model, X_test_torch)

    mse  = np.mean((y_pred - y_true)**2)
    mae  = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0,1]

    # baseline: predict 0 return
    mse0  = np.mean((0.0 - y_true)**2)
    mae0  = np.mean(np.abs(y_true))
    rmse0 = np.sqrt(mse0)

    # direction accuracy & confusion-style
    print_confusion_like(y_true, y_pred, thr=PROBA_THRESHOLD)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred - PROBA_THRESHOLD))
    dir_acc = float(dir_acc)

    # tiny PnL
    pnl_curve = tiny_pnl(y_true, y_pred, fee_bps=TAKER_FEE_BPS, thr=PROBA_THRESHOLD) if RUN_PNL else None

    print("\n=== FINAL TEST METRICS (log-return units) ===")
    print(f"MSE     : {mse:.6g}   (baseline 0: {mse0:.6g})")
    print(f"RMSE    : {rmse:.6g}  (baseline 0: {rmse0:.6g})   => ΔRMSE: {rmse0 - rmse:+.6g}")
    print(f"MAE     : {mae:.6g}   (baseline 0: {mae0:.6g})    => ΔMAE : {mae0 - mae:+.6g}")
    print(f"R^2     : {r2:.4f}")
    print(f"Corr    : {corr:.4f}")
    print(f"DirAcc  : {dir_acc*100:.2f}%")
    if TARGET_MODE == "return":
        print(f"MAE bp  : {mae*1e4:.1f} bp;  RMSE bp: {rmse*1e4:.1f} bp")
    if pnl_curve is not None:
        print(f"Final tiny-PnL (log units): {pnl_curve[-1]:.6g}  | fees={TAKER_FEE_BPS} bp")

    save_plots_and_csv("test", y_true, y_pred, pnl_curve=pnl_curve)
    print("Saved artifacts to ./artifacts/  (scatter, error hist, pnl, preds.csv)")
