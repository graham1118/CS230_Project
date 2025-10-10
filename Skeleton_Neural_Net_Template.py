import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib
import os
import pickle
from matplotlib import pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("USING DEVICE:", device)

# Save an object to a file specified at filepath
# This is done using pickle
def save_object_to_file(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
        
# Read an object from a specified filepath
# Note that the filepath needs to be saved to by pickle
# originally via the save_object_to_file functio
def read_object_from_file(filepath):
    with open(filepath, "rb") as f:
        out = pickle.load(f)
    return out

# Seeds all relevant components and makes everything deterministinc for reproducibility
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_split(n_train=3_000_000, n_test=1_000_000, d=50, seed=0):
    rng = np.random.default_rng(seed)

    def make(n):
        X = rng.standard_normal((n, d)).astype(np.float32)
        # 5th smallest (idx=4) and 5th largest (idx=d-5) via partial partition
        P = np.partition(X, (4, d-5), axis=1)
        fifth_small = P[:, 4]
        fifth_large = P[:, d-5]
        y = 0.5 * X.mean(axis=1) + 0.5 * (fifth_small + fifth_large) + np.random.randn(*(X.mean(axis = -1).shape))*0.3
        return X, y.astype(np.float32)

    X_train, Y_train = make(n_train)
    X_test,  Y_test  = make(n_test)
    return (X_train, Y_train), (X_test, Y_test)

class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout):
        super().__init__()

        assert layer_sizes[-1] == 1
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            l_1 = layer_sizes[i]
            l_2 = layer_sizes[i+1]
            layers.append(nn.Linear(l_1, l_2))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)

def loss_func(z, y):
    assert z.size() == y.size()
    return torch.mean(torch.square(z - y))

SEED = 0
results_save_path = "exp_results.pkl"
plot_save_path_suffix = "_loss_plot.jpg"
EPOCHS = 100

(X_train_np, Y_train_np), (X_test_np, Y_test_np) = make_split(
    300_000,
    100_000,
    50,
    SEED
)
seed_everything(SEED)

X_train_torch = torch.from_numpy(X_train_np).float()
Y_train_torch = torch.from_numpy(Y_train_np).float()
X_test_torch = torch.from_numpy(X_test_np).float()
Y_test_torch = torch.from_numpy(Y_test_np).float()

def run_full_train(
        layer_sizes, 
        dropout, 
        learning_rate, 
        weight_decay, 
        batch_size, 
        train_amt,
        epochs, 
        verbose_freq = 1
    ):
    MODEL = MLP(layer_sizes, dropout).to(device)
    optimizer = torch.optim.Adam(params = MODEL.parameters(), lr = learning_rate, weight_decay = weight_decay)

    print("Currently running Model with following architecture")
    print(MODEL)
    print("Currently running Optimizer with following params")
    print(optimizer)
    print("Using batch size:", batch_size)
    print("Total Epochs:", epochs)
    
    train_loss_hist = []
    test_loss_hist = []

    X_test = X_test_torch.to(device)
    Y_test = Y_test_torch.to(device)
    X_train_use = X_train_torch[:train_amt]
    Y_train_use = Y_train_torch[:train_amt]
    
    for epoch in range(epochs):
        train_idx_order = torch.randperm(X_train_use.size(0))
        X_train = X_train_use[train_idx_order].to(device)
        Y_train = Y_train_use[train_idx_order].to(device)

        MODEL.train()
        epoch_train_loss = 0
        for batch_idx in range(0,len(X_train),batch_size):
            X = X_train[batch_idx:batch_idx+batch_size]
            Y = Y_train[batch_idx:batch_idx+batch_size]

            optimizer.zero_grad()
            Z = MODEL(X)
            loss = loss_func(Z, Y)
            loss.backward()
            optimizer.step()
        
            epoch_train_loss += (loss.item() * len(X))
        
        epoch_train_loss /= len(X_train)
        train_loss_hist.append(epoch_train_loss)

        MODEL.eval()
        epoch_test_loss = 0
        for batch_idx in range(0,len(X_test),batch_size):
            X = X_test[batch_idx:batch_idx+batch_size]
            Y = Y_test[batch_idx:batch_idx+batch_size]

            with torch.no_grad():
                Z = MODEL(X)
                loss = loss_func(Z, Y)
                
            epoch_test_loss += (loss.item() * len(X))
                
        epoch_test_loss /= len(X_test)
        test_loss_hist.append(epoch_test_loss)

        if verbose_freq is not None and epoch%verbose_freq == 0:
            print("Epoch:", epoch, "Epoch Train Loss:", epoch_train_loss, "Epoch Test Loss:", epoch_test_loss)

    return train_loss_hist, test_loss_hist


all_exp_params = {
    "default" : {
        "layer_sizes" : [50, 100, 100, 1],
        "dropout" : 0,
        "weight_decay" : 0,
        "learning_rate" : 1e-4,
        "batch_size" : 256,
        "train_amt" : 10_000,
        "epochs" : EPOCHS
    },
}

exp_results = {}

for exp_tag, exp_params in all_exp_params.items():
    results = run_full_train(**exp_params)
    exp_results[exp_tag] = results

save_object_to_file(exp_results, results_save_path)
exp_results = read_object_from_file(results_save_path)

for exp_tag, exp_results in exp_results.items():
    exp_train_losses, exp_test_losses = exp_results
    plot_save_path = exp_tag + plot_save_path_suffix
    plt.clf()
    plt.plot(range(len(exp_train_losses)), exp_train_losses, label = "Train Loss")
    plt.plot(range(len(exp_test_losses)), exp_test_losses, label = "Test Loss")
    plt.title("Model Loss vs Epochs")
    plt.xlabel("Trained Epoch Count")
    plt.ylabel("Loss Value")
    plt.legend(loc = "upper right")
    plt.savefig(plot_save_path)
    plt.clf()

