import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from train_nnue_v2 import NNUEDataset, clustered_split

NUM_BUCKETS = 8


def compute_bucket_index(active_ids, num_buckets=NUM_BUCKETS):
    """Estimated-turn-count bucket for a position, derived from its active
    feature ids. Channel layout (channel = id // 64) matches both
    MigoyugoState::get_observation() and MigoyugoLightState's NNUEUpdate
    encoding: 0=OUR_MIGO, 1=OUR_YUGO, 2=OPPONENT_MIGO, 3=OPPONENT_YUGO.
    Each migo costs ~1 turn, each yugo ~4 (it consumes 3 migos + 1 placement).
    This exact formula must be kept in sync with the C++ player's
    compute_bucket_index (nnue/include/nnue/nnue_layerstacks_player.hpp).
    """
    migo = 0
    yugo = 0
    for fid in active_ids:
        channel = fid // 64
        if channel in (0, 2):
            migo += 1
        else:
            yugo += 1
    turns = min(migo + 4 * yugo, 80)
    return min(turns // 10, num_buckets - 1)


class BucketedDataset(Dataset):
    def __init__(self, base_dataset, bucket_indices):
        self.base_dataset = base_dataset
        self.bucket_indices = bucket_indices

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        feat_vec, score = self.base_dataset[idx]
        return feat_vec, torch.tensor(self.bucket_indices[idx], dtype=torch.long), score


class NNUELayerStacks(nn.Module):
    def __init__(self, num_buckets=NUM_BUCKETS):
        super().__init__()
        self.num_buckets = num_buckets
        # Shared feature transformer (accumulator), same as the single-net NNUE.
        self.l1 = nn.Linear(256, 256)
        # Per-bucket small heads ("experts") - each a full 256->16->32->1 stack.
        self.l2 = nn.ModuleList([nn.Linear(256, 16) for _ in range(num_buckets)])
        self.l3 = nn.ModuleList([nn.Linear(16, 32) for _ in range(num_buckets)])
        self.output = nn.ModuleList([nn.Linear(32, 1) for _ in range(num_buckets)])

    def forward(self, x, bucket_idx):
        h1 = torch.clamp(self.l1(x), 0.0, 1.0)

        out = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        for b in range(self.num_buckets):
            mask = bucket_idx == b
            if not mask.any():
                continue
            hb = torch.clamp(self.l2[b](h1[mask]), 0.0, 1.0)
            hb = torch.clamp(self.l3[b](hb), 0.0, 1.0)
            out[mask] = self.output[b](hb)
        return out


def print_bucket_histogram(bucket_indices, num_buckets=NUM_BUCKETS):
    counts = [0] * num_buckets
    for b in bucket_indices:
        counts[b] += 1
    print("[buckets] sample counts per bucket (estimated turns, ~10 per bucket):")
    for b, count in enumerate(counts):
        print(f"  bucket {b} (turns {b*10}-{b*10+9}{'+' if b == num_buckets - 1 else ''}): {count}")


def build_optimizer(name, params, lr, weight_decay):
    if name == "adamw":
        return optim.AdamW(params, lr=lr if lr is not None else 1e-3, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(params, lr=lr if lr is not None else 1e-2, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def main():
    parser = argparse.ArgumentParser(description="Layer-stacked (bucketed multi-expert) NNUE trainer")
    parser.add_argument("--data", default="training_data_mcts.bin")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=None,
                         help="Defaults to 1e-3 for adamw, 1e-2 for sgd")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.3)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", default="nnue_layerstacks_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNUELayerStacks().to(device)

    full_dataset = NNUEDataset(args.data)
    bucket_indices = [compute_bucket_index(fk) for fk in full_dataset.feature_keys]
    print_bucket_histogram(bucket_indices)

    bucketed_dataset = BucketedDataset(full_dataset, bucket_indices)

    train_indices, val_indices = clustered_split(full_dataset, args.group_size, args.val_split)
    train_dataset = Subset(bucketed_dataset, train_indices)
    val_dataset = Subset(bucketed_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    criterion = nn.MSELoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print(f"Training on {device} with optimizer={args.optimizer}...")

    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for features, bucket_idx, targets in train_loader:
            features = features.to(device)
            bucket_idx = bucket_idx.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features, bucket_idx)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Hard Constraint: Clamp weights after step to prevent C++ overflow
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(-1.9, 1.9)

            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, bucket_idx, targets in val_loader:
                features = features.to(device)
                bucket_idx = bucket_idx.to(device)
                targets = targets.to(device)
                outputs = model(features, bucket_idx)
                total_val_loss += criterion(outputs, targets).item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1} | Train Loss: {total_train_loss/len(train_loader):.6f} "
              f"| Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), args.output)
            print("  --> Model Saved (New Best)")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement > args.patience:
                print(f"Early stopping: no val improvement for {args.patience} epochs.")
                break

    print("Training finished.")


if __name__ == "__main__":
    main()
