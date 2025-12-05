import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import pickle
import time
import os
import sys
import matplotlib.pyplot as plt

# ============================================================
# Configuration & Hyperparameters
# ============================================================
# Paths (using relative paths assuming we are in Directional_Loss_Experiment/)
NPZ_PATH = "../Data Processing/preprocessed_data.npz" 
COL_NAMES_PATH = "../Data Processing/column_names.csv"
MU_SIGMA_PATH = "../Data Processing/mu_sigma_df.csv"
SAVE_PATH = "directional_model.pth"
LOG_PATH = "training_log.txt"

SEED = 42
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Model Params
MODEL_TYPE = 'EncoderGRU'  # 'GRU', 'LSTM', 'EncoderGRU'
GRU_HIDDEN_SIZE = 64
LATENT_SIZE = 32
GRU_OUTPUT_SIZE = 1
NUM_GRU_LAYERS = 1
DROPOUT = 0.2

# Training Params
NUM_EPOCHS = 40           # Reduced slightly for faster experimentation
BATCH_SIZE = 256          # Will be overwritten by data loader
INIT_LEARNING_RATE = 1e-3 # Slightly higher start for OneCycle
MAX_LR = 5e-3
WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM = 1.0

# === 核心改动：方向性 Loss 参数 ===
ALPHA_DIRECTION = 10.0    # 惩罚系数：方向错误时 Loss 放大 (1 + ALPHA) 倍

# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ============================================================
# Data Loading
# ============================================================
def load_data():
    print(f"[INFO] Loading data from {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    
    X_train = torch.from_numpy(data["X_train_batches"]).float().to(device)
    Y_train = torch.from_numpy(data["Y_train_batches"]).float().to(device)
    X_val   = torch.from_numpy(data["X_val_batches"]).float().to(device)
    Y_val   = torch.from_numpy(data["Y_val_batches"]).float().to(device)
    X_test  = torch.from_numpy(data["X_test_batches"]).float().to(device)
    Y_test  = torch.from_numpy(data["Y_test_batches"]).float().to(device)
    
    # Load Column Info
    with open(COL_NAMES_PATH, "r") as f:
        txt = f.read()
    feature_names = [x.strip().strip("[]'\"") for x in txt.replace("\n", ",").split(",") if x.strip()]
    close_idx = feature_names.index("log_return_close")
    
    # Load Stats
    _ms = pd.read_csv(MU_SIGMA_PATH)
    mu = torch.tensor(_ms["mu"].to_numpy(), dtype=torch.float32, device=device)
    sigma = torch.tensor(_ms["sigma"].to_numpy(), dtype=torch.float32, device=device)
    
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test), close_idx, (mu, sigma)

# ============================================================
# Models
# ============================================================
class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, latent_size=64):
        super(EncoderGRU, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, F = x.shape
        x_enc = self.encoder(x.view(B*T, F)).view(B, T, -1)
        out, _ = self.gru(x_enc)
        out = self.norm(out[:, -1, :])
        return self.fc(out)

# ============================================================
# Custom Loss Function (The Key Experiment)
# ============================================================
def direction_weighted_mse(pred, target, alpha=ALPHA_DIRECTION):
    """
    MSE + Directional Penalty
    Loss = MSE * (1 + alpha) if sign(pred) != sign(target)
           MSE               if sign(pred) == sign(target)
    """
    mse = (pred - target) ** 2
    
    # 这里的 target 应该是真实的 residual (Y - last_X)
    # 但为了简化，我们假设 target 已经是我们要拟合的目标（residual）
    # 只有当真实涨跌方向（target的符号）和预测方向（pred的符号）不一致时才惩罚
    
    # 注意：如果 target 是 0，sign 为 0，这里可能会有一些边界情况，但影响不大
    sign_match = (torch.sign(pred) == torch.sign(target))
    
    # 构造惩罚系数矩阵
    penalty = torch.where(sign_match, 
                          torch.tensor(1.0, device=pred.device), 
                          torch.tensor(1.0 + alpha, device=pred.device))
    
    loss = torch.mean(mse * penalty)
    return loss

# ============================================================
# Training Loop
# ============================================================
def compute_metrics(model, X, Y, close_idx):
    model.eval()
    total_loss = 0
    correct_dir = 0
    total_samples = 0
    
    with torch.no_grad():
        for xb, yb in zip(X, Y):
            # Predict Residual
            pred_resid = model(xb)
            last_close = xb[:, -1, close_idx].unsqueeze(-1)
            target_resid = yb[:, :, 0] - last_close if yb.dim() == 3 else yb - last_close
            
            # Loss (using standard MSE for fair validation comparison, or use custom?)
            # 通常 Validation 指标看 MSE 和 Accuracy，不需要带惩罚的 Loss
            loss = F.mse_loss(pred_resid, target_resid)
            total_loss += loss.item() * xb.size(0)
            
            # Directional Accuracy (on actual return, i.e. residual + last - last = residual)
            # Pred direction vs Target direction
            # Target residual represents the change from t-1 to t+H
            
            pred_sign = torch.sign(pred_resid)
            true_sign = torch.sign(target_resid)
            
            mask = (true_sign != 0)
            correct_dir += (pred_sign[mask] == true_sign[mask]).sum().item()
            total_samples += mask.sum().item() # Ignore zero-change targets
            
    avg_loss = total_loss / (X.shape[0] * X.shape[1])
    acc = correct_dir / total_samples if total_samples > 0 else 0
    return avg_loss, acc

def train():
    seed_everything(SEED)
    
    # 1. Load Data
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), close_idx, _ = load_data()
    
    # 2. Setup Model
    input_size = X_train.shape[-1]
    model = EncoderGRU(input_size, GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, NUM_GRU_LAYERS, LATENT_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    steps_per_epoch = len(X_train)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS,
        pct_start=0.3, div_factor=25.0, final_div_factor=1e4
    )
    
    print(f"[INFO] Model: {MODEL_TYPE} | Loss Alpha: {ALPHA_DIRECTION}")
    print(f"[INFO] Training for {NUM_EPOCHS} epochs...")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Shuffle batches
        perm = torch.randperm(len(X_train))
        X_train_cur = X_train[perm]
        Y_train_cur = Y_train[perm]
        
        for xb, yb in zip(X_train_cur, Y_train_cur):
            optimizer.zero_grad()
            
            # Predict
            pred_resid = model(xb)
            
            # Target Construction: We predict Residual (Y - Last_X)
            last_close = xb[:, -1, close_idx].unsqueeze(-1)
            
            # Handle Y shape (if it's [B, T, F] take close col, if [B, 1] take it directly)
            # Based on GRU.py logic, Y might be [B, 1] or [B, H, F]
            if yb.dim() == 3:
                target_val = yb[:, 0, close_idx].unsqueeze(-1) # Take first horizon step? Or last? 
                # Assuming Horizon=2, Y_batches usually stored the target at H.
                # Let's stick to what standard GRU.py does:
                # It seems Y_train_batches in npz is already the target value?
                # Let's check shape. Usually [B, 1] if flatten_features was used, or [B, H, F]
                target_val = yb[:, -1, 0].unsqueeze(-1) # Taking last time step, first feature (close)
            else:
                target_val = yb # Assuming [B, 1]
                
            target_resid = target_val - last_close
            
            # === CUSTOM LOSS CALCULATION ===
            loss = direction_weighted_mse(pred_resid, target_resid, alpha=ALPHA_DIRECTION)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(X_train)
        
        # Validation
        val_mse, val_acc = compute_metrics(model, X_val, Y_val, close_idx)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_mse)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d} | Train Loss (W): {avg_train_loss:.6f} | Val MSE: {val_mse:.6f} | Val DirAcc: {val_acc*100:.2f}%")
        
        # Save Best (Based on MSE or Acc? Usually MSE for stability, but we care about Acc)
        # Let's save based on Acc this time? Or sticking to MSE?
        # Let's stick to MSE for 'best model' definition to avoid overfitting to noise
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {'alpha': ALPHA_DIRECTION, 'model': MODEL_TYPE}
            }, SAVE_PATH)
            
    total_time = time.time() - start_time
    print(f"\n[DONE] Training finished in {total_time:.1f}s")
    print(f"Best Val MSE: {best_val_loss:.6f}")
    
    # Test Set Evaluation
    model.load_state_dict(torch.load(SAVE_PATH)['model_state_dict'])
    test_mse, test_acc = compute_metrics(model, X_test, Y_test, close_idx)
    print(f"\n>>> TEST RESULTS <<<")
    print(f"MSE: {test_mse:.6f}")
    print(f"Directional Accuracy: {test_acc*100:.2f}%")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.title(f'Directional Accuracy over Epochs (Alpha={ALPHA_DIRECTION})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('direction_acc_curve.png')
    print("Saved direction_acc_curve.png")

if __name__ == "__main__":
    train()

