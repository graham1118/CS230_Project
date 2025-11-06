# GRU.py:

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
from matplotlib import pyplot as plt
import os
import sys
from typing import Tuple
import time


# ============================================================
# Configuration
# ============================================================
results_save_path = "exp_results.pkl"
plot_save_path_suffix = "_loss_plot.jpg"

COMBINE_DATA = False
BLOCK_SHUFFLE = False

NPZ_FNAME = fname = f"preprocessed_data{'_COMBINED' if COMBINE_DATA else ''}{'_SHUFFLED' if BLOCK_SHUFFLE else ''}.npz"

NPZ_PATH = f"../Data Processing/{NPZ_FNAME}"  # produced by your preprocessing script
COL_NAMES_PATH = "../Data Processing/column_names.csv"
MU_SIGMA_PATH = "../Data Processing/mu_sigma_df.csv"
SAVE_PATH = "cur_best_model.pth"
SEED = 0 

# ============================================================
# Hyperparameters 
# ============================================================
"""
Try:
1) Different SEQ_LENs 
2) Different HORIZONs
3) Different NUM_GRU_LAYERS
4) Different LEARNING_RATEs
"""

INIT_LEARNING_RATE = 2e-4          # smaller LR helps stability
NUM_EPOCHS = 60
NUM_GRU_LAYERS = 1
GRU_HIDDEN_SIZE = 64
LATENT_SIZE = 32
CLIP_GRAD_NORM = 1.0    # gradient clipping to prevent explosion
MAX_LR = 2e-3
MODEL_TO_USE = 'EncoderGRU' # 'GRU' or 'LSTM' or 'EncoderGRU'
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2 #note used
USE_SCHEDULER = True    # ReduceLROnPlateau on val loss



# ----------------------------
# Kernel, Files & Reproducibility
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

def seed_everything(seed):   #ensures randomness is fixed across runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def compute_extras(model, X, Y):
    model.eval()
    preds = []
    trues = []
    for xb, yb in zip(X, Y):
        pr_resid = model(xb)
        pr = pr_resid + xb[:, -1, close_idx].unsqueeze(-1)
        # de-normalize close log-returns
        pr = (pr[:, 0] * sigma[close_idx] + mu[close_idx]).detach().cpu()
        yb = (yb[:, 0] * sigma[close_idx] + mu[close_idx]).detach().cpu()
        preds.append(pr); trues.append(yb)
    p = torch.cat(preds); t = torch.cat(trues)
    p_mean, t_mean = p.mean(), t.mean()
    p_std,  t_std  = p.std(unbiased=False), t.std(unbiased=False)
    cov = ((p - p_mean) * (t - t_mean)).mean()
    corr = float(cov / (p_std * t_std + 1e-12))
    mse = float(((p - t) ** 2).mean())
    t_var = float(t.var(unbiased=False))  # population variance of true raw returns
    r2 = float(1.0 - (mse / (t_var + 1e-12)))

    return {
        "pearson_r": corr,
        "pred_std": float(p_std),
        "true_std": float(t_std),
        "pred_mean": float(p_mean),
        "true_mean": float(t_mean),
        "mse_raw": mse,
        "r2_like": r2
    }


@torch.no_grad()
def baseline_metrics_persistence(X, Y):
    # predict that next return equals last observed close return in the sequence
    correct = total = 0
    abs_sum = sq_sum = 0.0
    for xb, yb in zip(X, Y):
        last_close = xb[:, -1, close_idx] * sigma[close_idx] + mu[close_idx]
        true_raw   = yb[:, 0] * sigma[close_idx] + mu[close_idx]
        pred_raw   = last_close
        # MSE/MAE on raw
        abs_sum += float((pred_raw - true_raw).abs().sum().item())
        sq_sum  += float(((pred_raw - true_raw)**2).sum().item())
        # sign acc
        mask = (true_raw != 0)
        correct += int((torch.sign(pred_raw[mask]) == torch.sign(true_raw[mask])).sum().item())
        total   += int(mask.sum().item())
    n = X.shape[0] * X.shape[1]
    return abs_sum/n, (sq_sum/n)**0.5, correct/max(1,total)

@torch.no_grad()
def baseline_metrics_zero(Y):
    # predict zero return
    abs_sum = sq_sum = 0.0
    correct = total = 0
    for yb in Y:
        true_raw = yb[:, 0] * sigma[close_idx] + mu[close_idx]
        pred_raw = torch.zeros_like(true_raw)
        abs_sum += float((pred_raw - true_raw).abs().sum().item())
        sq_sum  += float(((pred_raw - true_raw)**2).sum().item())
        mask = (true_raw != 0)
        correct += int((torch.sign(pred_raw[mask]) == torch.sign(true_raw[mask])).sum().item())
        total   += int(mask.sum().item())
    n = Y.shape[0] * Y.shape[1]
    return abs_sum/n, (sq_sum/n)**0.5, correct/max(1,total)




# ============================================
# Load & reshape preprocessed batches from NPZ
# ============================================
def load_preprocessed_data(path=NPZ_PATH):
    """
    Returns:
        dict: Dictionary containing all arrays stored in the file.
              Keys: 'X_train_batches', 'Y_train_batches', etc.
    """

    with np.load(path) as data:
        dataset = {key: data[key] for key in data.files}
    return dataset
def flatten_features_to_close(Y):
    return  Y [..., 0].unsqueeze(-1)


data = load_preprocessed_data()
with open(COL_NAMES_PATH, "r", encoding="utf-8", errors="ignore") as f:
    txt = f.read()
feature_names = [x.strip().strip("[]'\"") for x in txt.replace("\n", ",").split(",") if x.strip()]

# --- load normalization stats and close index ---
_ms = pd.read_csv(MU_SIGMA_PATH)
mu = torch.tensor(_ms["mu"].to_numpy(), dtype=torch.float32, device=device)
sigma = torch.tensor(_ms["sigma"].to_numpy(), dtype=torch.float32, device=device)
close_idx = feature_names.index("log_return_close")

print("[Sanity] close_idx =", close_idx)
print("[Sanity] mu_close, sigma_close =", float(mu[close_idx]), float(sigma[close_idx]))

X_train_batches = data["X_train_batches"]
Y_train_batches = data["Y_train_batches"]

X_val_batches = data["X_val_batches"]
Y_val_batches = data["Y_val_batches"]

X_test_batches = data["X_test_batches"]
Y_test_batches = data["Y_test_batches"]

print("[INFO] Converting data to torch tensors and moving to device...")
X_train_batches = torch.from_numpy(X_train_batches).to(torch.float32).to(device)
Y_train_batches = torch.from_numpy(Y_train_batches).to(torch.float32).to(device)
X_val_batches = torch.from_numpy(X_val_batches).to(torch.float32).to(device)
Y_val_batches   = torch.from_numpy(Y_val_batches).to(torch.float32).to(device)
X_test_batches  = torch.from_numpy(X_test_batches).to(torch.float32).to(device)
Y_test_batches  = torch.from_numpy(Y_test_batches).to(torch.float32).to(device)



#set target to be only close price, rather than all features
# Y_train_batches = flatten_features_to_close(Y_train_batches)
# Y_val_batches = flatten_features_to_close(Y_val_batches)
# Y_test_batches = flatten_features_to_close(Y_test_batches)


num_train_batches = X_train_batches.shape[0]
num_val_batches = X_val_batches.shape[0]
num_test_batches = X_test_batches.shape[0]
BATCH_SIZE = X_train_batches.shape[1]  #assumes all batches have same size
SEQ_LEN = X_train_batches.shape[2]
NUM_FEATURES = X_train_batches.shape[3]
GRU_OUTPUT_SIZE = 1


print("############### META INFO ####################")
print("features: ", feature_names, '\n\n')
print("(num_batches, batch_size, seq_len, num_features):")
print("Train batches:", X_train_batches.shape)
print("Validation batches:", X_val_batches.shape)
print("Test batches:", X_test_batches.shape)
print("Training Y shape:", Y_train_batches.shape)
print(f"batch_size = {BATCH_SIZE}, seq_len = {SEQ_LEN}, num_features = {NUM_FEATURES}")
print("##############################################\n\n")

# print("######## SANITY CHECK: CONFIRM NORMS #########")
# # Flatten across batch and batch_size dimensions
# X_flat = X_train_batches.reshape(-1, X_train_batches.shape[2], X_train_batches.shape[3])  # shape: (num_batches*batch_size, seq_len, num_features)
# X_unique = X_flat.reshape(-1, X_train_batches.shape[3])  # shape: (total_timesteps, num_features)
# feature_means = np.mean(X_unique, axis=0)
# feature_stds = np.std(X_unique, axis=0)
# print("feature means", [f"{x:.3f}" for x in feature_means])
# print("feature_stds", [f"{x:.3f}" for x in feature_stds])
# print("these won't be exactly 0 or 1 because they are only for one batch, not all of them")
# print("##############################################\n\n")


############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################



# ==================================
#  GRU for regression (scalar)
# ==================================

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)                    # [B, T, H]
        out = out[:, -1, :]                     # last timestep [B, H]
        out = self.norm(out)
        out = self.fc(out)
        return out

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, latent_size=64):
        """
        Drop-in replacement for your GRU model, but with an
        encoder in front that learns per-timestep features.

        Args:
            input_size:  feature dimension of each timestep
            hidden_size: GRU hidden size
            output_size: final output dimension
            num_layers:  number of GRU layers
            latent_size: encoder output dimension per timestep
        """
        super(EncoderGRU, self).__init__()

        # ---- Encoder (per-timestep feedforward feature extractor) ----
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
            nn.ReLU()
        )

        # ---- GRU (same structure as your baseline) ----
        self.gru = nn.GRU(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # ---- Normalization and output projection ----
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: [B, T, input_size]
        returns: [B, output_size]
        """
        B, T, F = x.shape

        # Encode each timestep independently
        x_enc = self.encoder(x.view(B*T, F))      # [B*T, latent_size]
        x_enc = x_enc.view(B, T, -1)              # [B, T, latent_size]

        # Temporal modeling with GRU
        out, _ = self.gru(x_enc)                  # [B, T, H]
        out = out[:, -1, :]                       # last timestep [B, H]

        # Normalize and project
        out = self.norm(out)
        out = self.fc(out)

        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=NUM_GRU_LAYERS):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out: (batch_size, seq_length, hidden_size)
        # (h_n, c_n) are the hidden and cell states
        out, (h_n, c_n) = self.lstm(x)

        # Take the output at the last timestep
        out = self.fc(out[:, -1, :])
        return out






# ==========================
# Evaluation Functions
# ==========================
    # compute_set_metrics() - call on full dataset after training
    # compute_set_loss()    - call on full dataset after each epoch

    # PyTorch funcions like y = model(x) and loss_func() are batch-level operations, so for 
    # set-level operations you need to write loops over batches and accumulate results.

@torch.no_grad()
def compute_set_metrics(model, X, Y, device="cpu"):
    model.eval()
    if isinstance(X, np.ndarray): X = torch.tensor(X, dtype=torch.float32)
    if isinstance(Y, np.ndarray): Y = torch.tensor(Y, dtype=torch.float32)
    X, Y = X.to(device), Y.to(device)

    abs_sum = 0.0
    sq_sum  = 0.0
    correct = 0
    total   = 0
    n_samples = 0

    for xb, yb in zip(X, Y):
        pr_resid = model(xb)
        pr = pr_resid + xb[:, -1, close_idx].unsqueeze(-1)


        abs_sum  += float((pr - yb).abs().sum().item())
        sq_sum   += float(((pr - yb) ** 2).sum().item())
        n_samples += pr.numel()

        # --- sign accuracy on *raw* (de-normalized) close log-returns ---
        pr_raw = pr[:, 0] * sigma[close_idx] + mu[close_idx]
        yb_raw = yb[:, 0] * sigma[close_idx] + mu[close_idx]
        pred_sign = torch.sign(pr_raw)
        true_sign = torch.sign(yb_raw)
        mask = (true_sign != 0)
        correct += int((pred_sign[mask] == true_sign[mask]).sum().item())
        total   += int(mask.sum().item())

    mae   = abs_sum / max(1, n_samples)
    rmse  = (sq_sum / max(1, n_samples)) ** 0.5
    sacc  = (correct / total) if total > 0 else float("nan")
    return mae, rmse, sacc

@torch.no_grad()
def compute_set_loss(model, X, Y, loss_func, device="cpu"):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    X, Y = X.to(device), Y.to(device)
    for xb, yb in zip(X, Y):
        pr_resid = model(xb)
        last = xb[:, -1, close_idx].unsqueeze(-1)
        pr_full = pr_resid + last
        loss = loss_func(pr_full, yb)
        total_loss += float(loss.item()) * xb.shape[0]
        num_samples += xb.shape[0]
    return total_loss / max(1, num_samples), total_loss

@torch.no_grad()
def compute_set_loss_close(model, X, Y, loss_func, device="cpu"):
    """
    Compute total and average loss over the dataset, 
    but only for the first feature of the model's output.
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    X, Y = X.to(device), Y.to(device)

    # Iterate over leading dimension (batches)
    for xb, yb in zip(X, Y):
        pr = model(xb)                    # Forward pass
        pr_first = pr[..., 0].unsqueeze(-1)  # keep only first feature
        yb_first = yb[..., 0].unsqueeze(-1)
        loss = loss_func(pr_first, yb_first)  # Compute batch loss
        total_loss += float(loss.item()) * xb.shape[0]
        num_samples += xb.shape[0]

    avg_loss = total_loss / max(1, num_samples)
    return avg_loss, total_loss


class EarlyStopper:
    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0

    def step(self, val):
        improved = val < (self.best - self.min_delta)
        if improved:
            self.best = val
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience


# ==========================
# Training / evaluation loop
# ==========================

def train_model(
    model,
    optimizer,
    loss_func,
    X_train,
    Y_train,
    X_val,
    Y_val,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    device="cpu",
    scheduler=None
):

    # --- Move model to device ---
    model = model.to(device)

    # --- Storage for losses ---
    train_losses = []
    val_losses = []

    USE_EARLY_STOP = False

    early = EarlyStopper(patience=15, min_delta=1e-3)
    best_state = None
    best_val = float("inf")

    # --- Epoch loop ---
    for epoch in range(num_epochs):
        model.train()
        epoch_t0 = time.time() 

        perm = torch.randperm(X_train.shape[0], device=device)
        X_train_epoch, Y_train_epoch = X_train[perm], Y_train[perm]

        PRINT_EVERY = 100

        for bi, (xb, yb) in enumerate(zip(X_train_epoch, Y_train_epoch), start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)

            if bi == 1:  # check first batch each epoch
                # inside the training loop, replace the dbg block with:
                with torch.no_grad():
                    last = xb[:, -1, close_idx].unsqueeze(-1)
                    out_std = pred.std().item()
                    tgt_std = (yb - last).std().item()
                    out_mean = pred.mean().item()
                    print(f"    [dbg] out_std={out_std:.4f} tgt_std={tgt_std:.4f} out_mean={out_mean:.4e}")


            last = xb[:, -1, close_idx].unsqueeze(-1)     # z-scored last close
            target_resid = yb - last
            loss = criterion(pred, target_resid)          # model predicts residual directly

            y_hat = pred
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # ← keep AFTER optimizer.step()
                

        CUTOFF = X_train.shape[0]

        # --- Compute average loss across batches for the epoch ---
        dt = time.time() - epoch_t0 
        train_loss, _ = compute_set_loss(model, X_train[:CUTOFF], Y_train[:CUTOFF], loss_func)
        val_loss, _ = compute_set_loss(model, X_val[:CUTOFF], Y_val[:CUTOFF], loss_func)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss={train_loss:.4f} | "
          f"Val Loss={val_loss:.4f} | time={dt:.1f}s")  
        for param_group in optimizer.param_groups:
            print(f"LR: {param_group['lr']:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if USE_EARLY_STOP:
            if early.step(val_loss):
                print(f"[EarlyStop] Stopped at epoch {epoch+1} with best val={best_val:.4f}")
                break

        if (epoch + 1) % 5 == 0:
            ex = compute_extras(model, X_val, Y_val)
            print(f"[Val/Extras@{epoch+1}] r={ex['pearson_r']:.3f} "
                f"pred_std={ex['pred_std']:.4f} true_std={ex['true_std']:.4f} "
                f"mse_raw={ex['mse_raw']:.2e}")


    
    # --- End of epoch loop ---
    mae_train, rmse_train, sign_acc_train = compute_set_metrics(model, X_train, Y_train)
    mae_val, rmse_val, sign_acc_val = compute_set_metrics(model, X_val, Y_val)
    train_metrics = (mae_train, rmse_train, sign_acc_train)
    val_metrics = (mae_val, rmse_val, sign_acc_val)

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Return history and final metrics ---
    return train_losses, val_losses, train_metrics, val_metrics



# ==========================
# Main: load, build, train
# ==========================
if __name__ == "__main__":
    seed_everything(SEED)
    print("[INFO] CWD:", os.getcwd())
    print("[INFO] Looking for NPZ at:", os.path.abspath(NPZ_PATH))


    if MODEL_TO_USE == 'GRU':
        model = GRU(input_size=X_train_batches.shape[-1], hidden_size=GRU_HIDDEN_SIZE, output_size=GRU_OUTPUT_SIZE, num_layers=NUM_GRU_LAYERS)
    if MODEL_TO_USE == "EncoderGRU":
        model = EncoderGRU(input_size=X_train_batches.shape[-1], hidden_size=GRU_HIDDEN_SIZE, output_size=GRU_OUTPUT_SIZE, num_layers=NUM_GRU_LAYERS, latent_size=LATENT_SIZE)
    elif MODEL_TO_USE == 'LSTM':
        model = LSTM(input_size=X_train_batches.shape[-1], hidden_size=GRU_HIDDEN_SIZE, output_size=GRU_OUTPUT_SIZE, num_layers=NUM_GRU_LAYERS)
    else: 
        print("[ERROR] invalid model to use")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = X_train_batches.shape[0]
    total_steps = steps_per_epoch * NUM_EPOCHS

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        total_steps=total_steps,
        pct_start=0.1,            # e.g., 40% of cycle increasing
        anneal_strategy='cos',    # cosine decrease after peak
        div_factor=10.0,          # initial LR = max_lr/div_factor
        final_div_factor=1e3      # final LR ~ max_lr/final_div_factor
    )




    # 5) Train
    print("[INFO] Starting training...")
    start = time.time()
    train_losses, val_losses, train_metrics, val_metrics = train_model(
        model,
        optimizer,
        criterion,
        X_train_batches,
        Y_train_batches,
        X_val_batches,
        Y_val_batches,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        device=device,
        scheduler=scheduler if USE_SCHEDULER else None
    )
    end = time.time()
    print(f"[INFO] Training time: {end - start:.2f} seconds")

    mae_train, rmse_train, sign_acc_train = train_metrics
    mae_val, rmse_val, sign_acc_val = val_metrics

    test_loss, _ = compute_set_loss(model, X_test_batches, Y_test_batches, criterion)
    mae_test, rmse_test, sign_acc_test = compute_set_metrics(model, X_test_batches, Y_test_batches)




    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, SAVE_PATH)

    print(f"✅ Model saved to {SAVE_PATH}")




    # -------------------- #
    # 2. Pretty print summary
    # -------------------- #
    print("\n" + "="*60)
    print("🚀 TRAINING COMPLETE — SUMMARY OF RESULTS")
    print("="*60)
    print(f"{'Dataset':<12}{'Loss':>10}{'MAE':>12}{'RMSE':>12}{'Sign Acc':>12}")
    print("-"*60)
    print(f"{'Train':<12}{train_losses[-1]:>10.4f}{mae_train:>12.4f}{rmse_train:>12.4f}{sign_acc_train:>12.3f}")
    print(f"{'Val':<12}{val_losses[-1]:>10.4f}{mae_val:>12.4f}{rmse_val:>12.4f}{sign_acc_val:>12.3f}")
    print(f"{'Test':<12}{test_loss:>10.4f}{mae_test:>12.4f}{rmse_test:>12.4f}{sign_acc_test:>12.3f}")
    extras_test = compute_extras(model, X_test_batches, Y_test_batches)
    print("[Extras/Test]", extras_test)

    mae_pers, rmse_pers, sacc_pers = baseline_metrics_persistence(X_test_batches, Y_test_batches)
    mae_zero, rmse_zero, sacc_zero = baseline_metrics_zero(Y_test_batches)
    print(f"[Baseline Persistence]  MAE={mae_pers:.4f} RMSE={rmse_pers:.4f} SignAcc={sacc_pers:.3f}")
    print(f"[Baseline Zero]         MAE={mae_zero:.4f} RMSE={rmse_zero:.4f} SignAcc={sacc_zero:.3f}")
    print("="*60 + "\n")

    with torch.no_grad():
        xb, yb = X_train_batches[0], Y_train_batches[0]
        pr = model(xb)
        print(f"[Sanity] pred_std={pr.std().item():.4f} target_std={yb.std().item():.4f}")

    


    # -------------------- #
    # 3. Plot training and validation loss
    # -------------------- #
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2, linestyle='--')
    plt.title("Training vs Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save + show
    save_path = os.path.join(os.getcwd(), "loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"[INFO] Loss curve saved to: {save_path}")
