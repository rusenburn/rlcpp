import argparse
import struct

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset


class NNUEDataset(Dataset):
    def __init__(self, filename):
        self.samples = []
        # Sorted tuple of active feature ids per record, kept alongside the
        # dense feat_vec purely so the split can detect duplicate positions
        # (two records with the same active-id tuple are the same board).
        self.feature_keys = []
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            while True:
                data = f.read(4)  # Score (float)
                if not data:
                    break
                score = struct.unpack("f", data)[0]

                data = f.read(2)  # Count (int16)
                count = struct.unpack("h", data)[0]

                data = f.read(2 * count)  # Feature IDs
                features = struct.unpack(f"{count}h", data)

                feat_vec = torch.zeros(256)
                active_ids = []
                for fid in features:
                    if 0 <= fid < 256:  # Safety check
                        feat_vec[fid] = 1.0
                        active_ids.append(fid)

                self.samples.append((feat_vec, torch.tensor([score])))
                self.feature_keys.append(tuple(sorted(active_ids)))
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        # Architecture: 256 -> 256 -> 16 -> 32 -> 1 (unchanged: the C++
        # quantized inference path hardcodes these widths)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 16)
        self.l3 = nn.Linear(16, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # Clipped ReLU [0, 1] matches the std::clamp(x, 0, 127) logic in C++
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        x = torch.clamp(self.l3(x), 0.0, 1.0)
        return self.output(x)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def clustered_split(dataset: NNUEDataset, group_size: int, val_split: float):
    """Splits the dataset into train/val indices without ever letting a
    symmetric sibling or a cross-game duplicate position straddle the
    boundary.

    Tier 1: consecutive records are grouped in chunks of `group_size` (the
    generator writes each position's 8 symmetric orientations back to back).
    Tier 2: any two groups that share an identical feature vector (the same
    board recurring in a different game) are unioned into one cluster.
    """
    n_samples = len(dataset)
    n_groups = n_samples // group_size
    leftover = n_samples % group_size
    if leftover:
        print(
            f"[split] Warning: {leftover} trailing records don't form a "
            f"full group of {group_size}; folding them into the last group."
        )
        n_groups += 1

    def group_of(idx):
        return min(idx // group_size, n_groups - 1)

    uf = UnionFind(n_groups)

    feature_key_to_groups = {}
    for idx, key in enumerate(dataset.feature_keys):
        gid = group_of(idx)
        groups_for_key = feature_key_to_groups.setdefault(key, set())
        groups_for_key.add(gid)

    for groups_for_key in feature_key_to_groups.values():
        groups_for_key = list(groups_for_key)
        for other in groups_for_key[1:]:
            uf.union(groups_for_key[0], other)

    clusters = {}
    for gid in range(n_groups):
        root = uf.find(gid)
        clusters.setdefault(root, []).append(gid)

    cluster_list = list(clusters.values())

    generator = torch.Generator().manual_seed(torch.initial_seed() & 0xFFFFFFFF)
    perm = torch.randperm(len(cluster_list), generator=generator).tolist()
    cluster_list = [cluster_list[i] for i in perm]

    n_val_target = int(round(val_split * n_samples))
    val_group_ids = set()
    val_count = 0
    for cluster in cluster_list:
        cluster_size = sum(
            (group_size if gid < n_groups - 1 or leftover == 0 else leftover)
            for gid in cluster
        )
        if val_count >= n_val_target:
            break
        val_group_ids.update(cluster)
        val_count += cluster_size

    train_indices = []
    val_indices = []
    for idx in range(n_samples):
        gid = group_of(idx)
        if gid in val_group_ids:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    n_merged_clusters = sum(1 for c in cluster_list if len(c) > 1)
    print(
        f"[split] {n_groups} groups -> {len(cluster_list)} clusters "
        f"({n_merged_clusters} contain a cross-game duplicate)."
    )
    print(f"[split] train={len(train_indices)} val={len(val_indices)}")

    return train_indices, val_indices


def build_optimizer(name, params, lr, weight_decay):
    if name == "adamw":
        return optim.AdamW(
            params, lr=lr if lr is not None else 1e-3, weight_decay=weight_decay
        )
    elif name == "sgd":
        return optim.SGD(
            params,
            lr=lr if lr is not None else 1e-2,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def main():
    parser = argparse.ArgumentParser(description="Regularized NNUE trainer (v2)")
    parser.add_argument("--data", default="training_data.bin")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Defaults to 1e-3 for adamw, 1e-2 for sgd",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.3)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", default="nnue_model_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNUE().to(device)

    full_dataset = NNUEDataset(args.data)
    train_indices, val_indices = clustered_split(
        full_dataset, args.group_size, args.val_split
    )
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    criterion = nn.MSELoss()
    optimizer = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print(f"Training on {device} with optimizer={args.optimizer}...")

    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Hard Constraint: Clamp weights after step to prevent C++ overflow
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(-1.9, 1.9)

            total_train_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                total_val_loss += criterion(outputs, targets).item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1} | Train Loss: {total_train_loss/len(train_loader):.6f} "
            f"| Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}"
        )

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
