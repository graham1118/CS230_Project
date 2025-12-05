import torch
import numpy as np
import pandas as pd
from train_directional import EncoderGRU, load_data, compute_metrics, GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, NUM_GRU_LAYERS, LATENT_SIZE

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = "directional_model.pth"

def check():
    # Load Data
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), close_idx, _ = load_data()
    
    # Load Model
    input_size = X_test.shape[-1]
    model = EncoderGRU(input_size, GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, NUM_GRU_LAYERS, LATENT_SIZE).to(device)
    
    try:
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {SAVE_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Compute Metrics
    train_mse, train_acc = compute_metrics(model, X_train, Y_train, close_idx)
    val_mse, val_acc = compute_metrics(model, X_val, Y_val, close_idx)
    test_mse, test_acc = compute_metrics(model, X_test, Y_test, close_idx)

    print("\n" + "="*40)
    print(f"RESULTS SUMMARY")
    print("="*40)
    print(f"Train Acc: {train_acc*100:.2f}% | MSE: {train_mse:.6f}")
    print(f"Val   Acc: {val_acc*100:.2f}% | MSE: {val_mse:.6f}")
    print(f"Test  Acc: {test_acc*100:.2f}% | MSE: {test_mse:.6f}")
    print("="*40)

    if test_acc > 0.52:
        print("Verdict: GOOD. Significant improvement over random guessing (50%).")
    elif test_acc > 0.50:
        print("Verdict: MARGINAL. Better than random, but barely.")
    else:
        print("Verdict: POOR. Not better than random.")

if __name__ == "__main__":
    check()

