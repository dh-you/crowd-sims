"""
Train a behavioral cloning policy on the collected crowd simulation data.
Run collect_data.py first to generate bc_data.json.

Install dependencies:
    pip install torch numpy
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
K_NEIGHBORS = 5
STATE_DIM = 4 + K_NEIGHBORS * 4   # 24
ACTION_DIM = 2                     # vel_x, vel_y
HIDDEN_DIM = 128
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
TRAIN_SPLIT = 0.9


# ── Dataset ──────────────────────────────────────────────────────────────────
class CrowdDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# ── Model ────────────────────────────────────────────────────────────────────
class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
with open("bc_data.json") as f:
    data = json.load(f)

states = np.array([d["state"] for d in data], dtype=np.float32)
actions = np.array([d["action"] for d in data], dtype=np.float32)
print(f"Loaded {len(states)} samples. State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")

# Normalize
state_mean = states.mean(axis=0)
state_std = states.std(axis=0) + 1e-8
action_mean = actions.mean(axis=0)
action_std = actions.std(axis=0) + 1e-8

states_norm = (states - state_mean) / state_std
actions_norm = (actions - action_mean) / action_std

# Save normalization stats for inference later
np.save("norm_stats.npy", {
    "state_mean": state_mean,
    "state_std": state_std,
    "action_mean": action_mean,
    "action_std": action_std
})

# Train/val split
n = len(states_norm)
split = int(n * TRAIN_SPLIT)
idx = np.random.permutation(n)
train_idx, val_idx = idx[:split], idx[split:]

train_ds = CrowdDataset(states_norm[train_idx], actions_norm[train_idx])
val_ds = CrowdDataset(states_norm[val_idx], actions_norm[val_idx])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")


# ── Train ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = BCPolicy(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for states_b, actions_b in train_loader:
        states_b, actions_b = states_b.to(device), actions_b.to(device)
        pred = model(states_b)
        loss = loss_fn(pred, actions_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(states_b)
    train_loss /= len(train_ds)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for states_b, actions_b in val_loader:
            states_b, actions_b = states_b.to(device), actions_b.to(device)
            pred = model(states_b)
            val_loss += loss_fn(pred, actions_b).item() * len(states_b)
    val_loss /= len(val_ds)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "bc_policy.pt")

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print(f"\nDone. Best val loss: {best_val_loss:.4f}. Model saved to bc_policy.pt")
print("\nNext step: run inference.py to see the BC policy control agents in the simulation.")
