import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Keep the Class definitions at the TOP level
class CrowdDataset(Dataset):
    def __init__(self, states, actions):
        self.states  = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class BCPolicy(nn.Module):
    """
    Tanh-bounded output in [-1, 1].

    Why tanh: at test time the policy sees states the expert never visited.
    A linear head will happily output velocities of 50+ pixels/frame on those
    states. Tanh caps it so the worst the policy can do is "move at the
    fastest speed in the training set", not "teleport".
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return torch.tanh(self.trunk(x))

# 2. Put all the execution logic inside this "Main Guard"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Scene name (e.g. airplane, queue)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden width. 64 is honest for ~6k samples; 128 for 100k+.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--stationary_keep", type=float, default=1.0,
                        help="Fraction of stationary frames to keep. 1.0 = keep all "
                             "(recommended for low-data regimes). 0.5 = drop half.")
    parser.add_argument("--stationary_threshold", type=float, default=0.5)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber",
                        help="Huber is more robust to the mixed stationary/moving "
                             "regime when the dataset is small.")
    args = parser.parse_args()

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.scene)
    data_path = os.path.join(DATA_DIR, "expert_data.npz")

    print(f"Loading {data_path}...")
    data = np.load(data_path)
    states = data["states"]
    actions = data["actions"]

    # --- DATA BALANCING ---
    # In the low-data regime stationary frames are signal, not noise: they
    # teach "stand still when blocked". Default keep=1.0 (keep them all).
    speeds = np.linalg.norm(actions, axis=1)
    moving_indices = np.where(speeds > args.stationary_threshold)[0]
    stationary_indices = np.where(speeds <= args.stationary_threshold)[0]

    np.random.seed(42)
    num_to_keep = int(len(stationary_indices) * args.stationary_keep)
    kept_stationary = np.random.choice(stationary_indices, num_to_keep, replace=False)

    final_indices = np.concatenate([moving_indices, kept_stationary])
    np.random.shuffle(final_indices)

    states = states[final_indices]
    actions = actions[final_indices]

    STATE_DIM = states.shape[1]
    ACTION_DIM = actions.shape[1]
    print(f"After balancing: {len(states)} samples "
          f"({len(moving_indices)} moving + {len(kept_stationary)} stationary). "
          f"STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}")

    # --- NORMALIZATION ---
    # State: zero-mean unit-var.
    # Action: scale by max absolute value (per-axis) so labels land *exactly*
    # in [-1, 1] with no clipping needed. The previous std-then-clip approach
    # squashed fast walks (|v| ~= MAXSPEED) down to 1.0 in norm space, which
    # de-normalized back to roughly std*1.0 = 2.0 px/frame instead of 5.0 —
    # i.e., agents could never reach top speed. max-abs preserves the full range.
    state_mean = states.mean(axis=0)
    state_std  = states.std(axis=0) + 1e-8
    action_mean = np.zeros_like(actions[0])  # don't center actions; preserves sign symmetry
    action_std  = np.abs(actions).max(axis=0) + 1e-8  # per-axis max magnitude

    states_norm  = (states  - state_mean)  / state_std
    actions_norm = (actions - action_mean) / action_std
    # No clip needed — by construction actions_norm is in [-1, 1].

    np.save(os.path.join(DATA_DIR, "norm_stats.npy"), {
        "state_mean":  state_mean,
        "state_std":   state_std,
        "action_mean": action_mean,
        "action_std":  action_std,
    })

    n = len(states_norm)
    split = int(n * 0.9)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = CrowdDataset(states_norm[train_idx], actions_norm[train_idx])
    val_ds   = CrowdDataset(states_norm[val_idx],   actions_norm[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch * 4,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BCPolicy(STATE_DIM, ACTION_DIM, args.hidden).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine-decay the LR to ~1% of initial over the run. Reduces output noise
    # in the late epochs: when the loss surface is nearly flat, a high LR keeps
    # the weights wiggling around the minimum and the policy outputs different
    # velocities on near-identical states (the "constant jitter" you saw).
    # Annealing locks the weights in tight.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    loss_fn = nn.SmoothL1Loss() if args.loss == "huber" else nn.MSELoss()
    print(f"Loss: {args.loss} | hidden={args.hidden} | wd={args.weight_decay} | "
          f"LR: {args.lr} -> {args.lr*0.01:.2e} (cosine)")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for s, a in train_loader:
            s, a = s.to(device, non_blocking=True), a.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(s), a)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(s)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, a in val_loader:
                s, a = s.to(device, non_blocking=True), a.to(device, non_blocking=True)
                val_loss += loss_fn(model(s), a).item() * len(s)
        val_loss /= len(val_ds)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "bc_policy.pt"))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}  ->  {args.scene}/bc_policy.pt")
