import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PHASE_WAITING = 0
PHASE_TO_AISLE = 1


class CrowdDataset(Dataset):
    def __init__(self, states, actions, phases):
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self.phases = torch.as_tensor(phases, dtype=torch.int64)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.phases[idx]


class BCPolicy(nn.Module):
    """The existing 5-input, tanh-bounded behavioral-cloning policy."""

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


def parse_seeds(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one integer seed")
    return seeds


def split_indices(episode_ids, sample_count, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    if episode_ids is not None:
        episodes = np.unique(episode_ids)
        if len(episodes) < 2:
            raise ValueError("episode-aware validation requires at least two complete episodes")
        rng.shuffle(episodes)
        val_count = max(1, int(round(len(episodes) * validation_fraction)))
        val_count = min(val_count, len(episodes) - 1)
        val_episodes = episodes[-val_count:]
        val_mask = np.isin(episode_ids, val_episodes)
        return np.flatnonzero(~val_mask), np.flatnonzero(val_mask), val_episodes

    indices = rng.permutation(sample_count)
    split = int(sample_count * (1.0 - validation_fraction))
    return indices[:split], indices[split:], None


def balance_phase_indices(indices, phases, samples_per_phase, seed):
    if np.all(phases[indices] < 0):
        return indices

    rng = np.random.default_rng(seed)
    groups = [indices[phases[indices] == phase] for phase in np.unique(phases[indices])]
    target = min(samples_per_phase, max(len(group) for group in groups))
    balanced = []
    for group in groups:
        balanced.append(rng.choice(group, target, replace=len(group) < target))
    result = np.concatenate(balanced)
    rng.shuffle(result)
    return result


def elementwise_loss(prediction, target, loss_name):
    if loss_name == "huber":
        return nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    return nn.functional.mse_loss(prediction, target, reduction="none")


def phase_weighted_loss(
    prediction,
    target,
    phases,
    loss_name,
    waiting_weight,
    aisle_vertical_weight,
):
    losses = elementwise_loss(prediction, target, loss_name)
    weights = torch.ones_like(losses)
    waiting = phases == PHASE_WAITING
    to_aisle = phases == PHASE_TO_AISLE
    weights[waiting] *= waiting_weight
    weights[to_aisle, 1] *= aisle_vertical_weight
    return (losses * weights).sum() / weights.sum()


def validation_metrics(model, loader, device, args):
    model.eval()
    weighted_total = 0.0
    sample_total = 0
    phase_predictions = {PHASE_WAITING: [], PHASE_TO_AISLE: []}
    with torch.no_grad():
        for states, actions, phases in loader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            phases = phases.to(device, non_blocking=True)
            predictions = model(states)
            loss = phase_weighted_loss(
                predictions,
                actions,
                phases,
                args.loss,
                args.waiting_weight,
                args.aisle_vertical_weight,
            )
            weighted_total += loss.item() * len(states)
            sample_total += len(states)
            for phase in phase_predictions:
                mask = phases == phase
                if mask.any():
                    phase_predictions[phase].append(predictions[mask].cpu())

    metrics = {"loss": weighted_total / sample_total}
    waiting_predictions = phase_predictions[PHASE_WAITING]
    if waiting_predictions:
        waiting = torch.cat(waiting_predictions)
        metrics["waiting_speed_norm"] = torch.linalg.vector_norm(waiting, dim=1).mean().item()
    aisle_predictions = phase_predictions[PHASE_TO_AISLE]
    if aisle_predictions:
        aisle = torch.cat(aisle_predictions)
        metrics["aisle_vertical_abs_norm"] = aisle[:, 1].abs().mean().item()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42"))
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--samples_per_phase", type=int, default=30000)
    parser.add_argument("--validation_samples_per_phase", type=int, default=20000)
    parser.add_argument("--waiting_weight", type=float, default=5.0)
    parser.add_argument("--aisle_vertical_weight", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--candidate", action="store_true",
                        help="Read and write candidate artifacts for gated promotion.")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.scene)
    suffix = ".candidate" if args.candidate else ""
    data_path = os.path.join(data_dir, f"expert_data{suffix}.npz")
    print(f"Loading {data_path}...")
    data = np.load(data_path)
    states = data["states"]
    actions = data["actions"]
    phases = data["phases"] if "phases" in data else np.full(len(states), -1, dtype=np.int8)
    episode_ids = data["episode_ids"] if "episode_ids" in data else None

    train_indices, val_indices, val_episodes = split_indices(
        episode_ids, len(states), args.validation_fraction, 42
    )
    if val_episodes is not None:
        train_episodes = np.unique(episode_ids[train_indices])
        if np.intersect1d(train_episodes, val_episodes).size:
            raise AssertionError("training and validation episodes overlap")
        print(f"Episode split: train={train_episodes.tolist()} val={val_episodes.tolist()}")

    # Statistics come only from training episodes, preventing validation leakage.
    train_states = states[train_indices]
    train_actions = actions[train_indices]
    state_mean = train_states.mean(axis=0)
    state_std = train_states.std(axis=0) + 1e-8
    action_mean = np.zeros_like(train_actions[0])
    action_std = np.abs(train_actions).max(axis=0) + 1e-8
    stats = {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    stats_path = os.path.join(data_dir, f"norm_stats{suffix}.npy")
    np.save(stats_path, stats)

    states_norm = (states - state_mean) / state_std
    actions_norm = (actions - action_mean) / action_std
    balanced_train_indices = balance_phase_indices(
        train_indices, phases, args.samples_per_phase, 42
    )
    balanced_val_indices = balance_phase_indices(
        val_indices, phases, args.validation_samples_per_phase, 43
    )
    phase_counts = {
        int(phase): int(np.sum(phases[balanced_train_indices] == phase))
        for phase in np.unique(phases[balanced_train_indices])
    }
    print(
        f"Train samples: {len(balanced_train_indices)} {phase_counts} | "
        f"validation samples: {len(balanced_val_indices)}"
    )

    train_dataset = CrowdDataset(
        states_norm[balanced_train_indices],
        actions_norm[balanced_train_indices],
        phases[balanced_train_indices],
    )
    val_dataset = CrowdDataset(
        states_norm[balanced_val_indices],
        actions_norm[balanced_val_indices],
        phases[balanced_val_indices],
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch * 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    best_overall_score = float("inf")
    best_overall_state = None
    best_seed = None

    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True,
            generator=generator,
        )
        model = BCPolicy(states.shape[1], actions.shape[1], args.hidden).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        best_seed_score = float("inf")
        best_seed_state = None
        stale_epochs = 0

        for epoch in range(args.epochs):
            model.train()
            train_total = 0.0
            for batch_states, batch_actions, batch_phases in train_loader:
                batch_states = batch_states.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)
                batch_phases = batch_phases.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = phase_weighted_loss(
                    model(batch_states),
                    batch_actions,
                    batch_phases,
                    args.loss,
                    args.waiting_weight,
                    args.aisle_vertical_weight,
                )
                loss.backward()
                optimizer.step()
                train_total += loss.item() * len(batch_states)
            scheduler.step()
            metrics = validation_metrics(model, val_loader, device, args)
            score = metrics["loss"]
            if score < best_seed_score:
                best_seed_score = score
                best_seed_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                extras = " ".join(
                    f"{key}={value:.5f}" for key, value in metrics.items() if key != "loss"
                )
                print(
                    f"seed={seed} epoch={epoch + 1}/{args.epochs} "
                    f"train={train_total / len(train_dataset):.5f} "
                    f"val={score:.5f} {extras}"
                )
            if stale_epochs >= args.patience:
                print(f"seed={seed} early stop at epoch {epoch + 1}")
                break

        if best_seed_score < best_overall_score:
            best_overall_score = best_seed_score
            best_overall_state = best_seed_state
            best_seed = seed

    output_path = os.path.join(data_dir, f"bc_policy{suffix}.pt")
    torch.save(best_overall_state, output_path)
    print(
        f"Best seed={best_seed}, weighted validation loss={best_overall_score:.6f} "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    main()
