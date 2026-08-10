"""Layer-stacked (bucketed multi-expert) NNUE trainer, 384-feature version.

Differences from train_nnue_layerstacks.py:

  * 384 inputs instead of 256. Channels 0-3 are unchanged (OUR_MIGO, OUR_YUGO,
    OPP_MIGO, OPP_YUGO); channels 4 and 5 are the piline sets - the empty
    squares each player may not play on because doing so would build an
    unbroken line of more than four. Produce the data with
    `convert_nnue_data_384 training_data_mcts.bin training_data_mcts_384.bin`.

  * compute_bucket_index no longer counts the piline channels. The old version
    had a bare `else` that swept channels 4 and 5 into the Yugo count, where
    they were multiplied by four.

  * The dataset stores feature ids sparsely and densifies per batch in a
    collate_fn. The old one materialised one dense tensor per record, which at
    384 inputs and a million records would be about 1.5 GB before overhead.

Usage:
    python train_nnue_layerstacks_v2.py --data training_data_mcts_384.bin
"""

import argparse
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from train_nnue_v2 import clustered_split

NUM_BUCKETS = 8
NUM_FEATURES = 384
L1_SIZE = 256

# Channel layout, mirroring games/include/games/migoyugo_bb.hpp.
CH_OUR_MIGO, CH_OUR_YUGO, CH_OPP_MIGO, CH_OPP_YUGO, CH_OUR_PILINE, CH_OPP_PILINE = range(6)
MIGO_CHANNELS = (CH_OUR_MIGO, CH_OPP_MIGO)
YUGO_CHANNELS = (CH_OUR_YUGO, CH_OPP_YUGO)


def compute_bucket_index(active_ids, num_buckets=NUM_BUCKETS):
    """Estimated-turn-count bucket for a position, from its active feature ids.

    Each Migo on the board cost about one turn and each Yugo about four (it
    consumes three Migos plus the placement). The piline channels are derived
    from the pieces rather than being pieces themselves, so they are ignored -
    counting them here is what the old bare `else` got wrong.

    Must stay in sync with NNUELayerStacksPlayerV2::compute_bucket_index in
    nnue/include/nnue/nnue_layerstacks_player_v2.hpp, which derives the same
    number by popcount over the piece bitboards.
    """
    migo = 0
    yugo = 0
    for fid in active_ids:
        channel = fid // 64
        if channel in MIGO_CHANNELS:
            migo += 1
        elif channel in YUGO_CHANNELS:
            yugo += 1
    turns = min(migo + 4 * yugo, 80)
    return min(turns // 10, num_buckets - 1)


class SparseNNUEDataset(Dataset):
    """Reads the binary training set, keeping the active feature ids sparse.

    Record format: float32 score, int16 count, int16 feature_id[count].
    Exposes `feature_keys` so clustered_split() can spot duplicate positions.
    """

    def __init__(self, filename, n_features=NUM_FEATURES):
        print(f"Loading data from {filename}...")
        self.n_features = n_features

        with open(filename, "rb") as f:
            raw = f.read()
        view = memoryview(raw)

        scores = []
        flat_ids = []
        offsets = [0]
        self.feature_keys = []

        pos = 0
        total = len(raw)
        while pos < total:
            (score,) = struct.unpack_from("<f", view, pos)
            pos += 4
            (count,) = struct.unpack_from("<h", view, pos)
            pos += 2
            ids = struct.unpack_from(f"<{count}h", view, pos)
            pos += 2 * count

            for fid in ids:
                if not 0 <= fid < n_features:
                    raise ValueError(
                        f"feature id {fid} is outside 0..{n_features - 1}. If this file "
                        "still holds 256-feature records, convert it first with "
                        "run/convert_nnue_data_384.cpp."
                    )

            scores.append(score)
            flat_ids.extend(ids)
            offsets.append(len(flat_ids))
            self.feature_keys.append(tuple(sorted(ids)))

        self.scores = np.asarray(scores, dtype=np.float32)
        self.flat_ids = np.asarray(flat_ids, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)

        self.bucket_indices = np.asarray(
            [compute_bucket_index(key) for key in self.feature_keys], dtype=np.int64
        )

        n_piline = int((self.flat_ids >= 256).sum())
        print(f"Loaded {len(self.scores)} samples.")
        print(
            f"  mean active features/record: {len(self.flat_ids) / max(len(self.scores), 1):.2f} "
            f"(piline {n_piline / max(len(self.scores), 1):.2f})"
        )
        if n_piline == 0:
            print(
                "  WARNING: no piline features at all. This file looks like it was "
                "never converted; the two new channels will train as dead inputs."
            )

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        lo, hi = self.offsets[idx], self.offsets[idx + 1]
        return self.flat_ids[lo:hi], self.bucket_indices[idx], self.scores[idx]


def collate_sparse(batch, n_features=NUM_FEATURES):
    """Densifies a batch of sparse records into the (features, bucket, target) triple."""
    size = len(batch)
    rows = np.concatenate([np.full(len(ids), i, dtype=np.int64) for i, (ids, _, _) in enumerate(batch)])
    cols = np.concatenate([ids for ids, _, _ in batch])

    features = torch.zeros(size, n_features)
    features[torch.from_numpy(rows), torch.from_numpy(cols)] = 1.0

    buckets = torch.as_tensor([b for _, b, _ in batch], dtype=torch.long)
    targets = torch.as_tensor([[s] for _, _, s in batch], dtype=torch.float32)
    return features, buckets, targets


class NNUELayerStacksV2(nn.Module):
    def __init__(self, num_buckets=NUM_BUCKETS, n_features=NUM_FEATURES):
        super().__init__()
        self.num_buckets = num_buckets
        # Shared feature transformer (the accumulator).
        self.l1 = nn.Linear(n_features, L1_SIZE)
        # Per-bucket experts, each a full 256 -> 16 -> 32 -> 1 stack.
        self.l2 = nn.ModuleList([nn.Linear(L1_SIZE, 16) for _ in range(num_buckets)])
        self.l3 = nn.ModuleList([nn.Linear(16, 32) for _ in range(num_buckets)])
        self.output = nn.ModuleList([nn.Linear(32, 1) for _ in range(num_buckets)])

    def forward(self, x, bucket_idx):
        # Clipped ReLU [0,1] matches the clamp(x, 0, 127) the C++ inference does.
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
    counts = np.bincount(np.asarray(bucket_indices), minlength=num_buckets)
    print("[buckets] sample counts per bucket (estimated turns, ~10 per bucket):")
    for b, count in enumerate(counts):
        suffix = "+" if b == num_buckets - 1 else ""
        print(f"  bucket {b} (turns {b*10}-{b*10+9}{suffix}): {count}")
    if (counts == 0).any():
        empty = [b for b, c in enumerate(counts) if c == 0]
        print(f"  WARNING: buckets {empty} have no samples; their experts stay at init.")


def build_optimizer(name, params, lr, weight_decay):
    if name == "adamw":
        return optim.AdamW(params, lr=lr if lr is not None else 1e-3, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr if lr is not None else 1e-2, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Layer-stacked NNUE trainer, 384-feature (piline) version"
    )
    parser.add_argument("--data", default="training_data_mcts_384.bin")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=None,
                        help="Defaults to 1e-3 for adamw, 1e-2 for sgd")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.3)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output", default="nnue_layerstacks_v2_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNUELayerStacksV2().to(device)

    dataset = SparseNNUEDataset(args.data)
    print_bucket_histogram(dataset.bucket_indices)

    train_indices, val_indices = clustered_split(dataset, args.group_size, args.val_split)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_sparse, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=collate_sparse, num_workers=args.num_workers)

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

            # Hard constraint: keep every weight inside what int16 quantization
            # at scale 128 can represent without overflowing in C++.
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
                total_val_loss += criterion(model(features, bucket_idx), targets).item()

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
    print(f"Next: python export_nnue_layerstacks_v2.py {args.output} "
          "../checkpoints/nnue_layerstacks_v2_weights.bin")


if __name__ == "__main__":
    main()
