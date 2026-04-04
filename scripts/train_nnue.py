import torch
import torch.nn as nn
import torch.optim as optim
import struct
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class NNUEDataset(Dataset):
    def __init__(self, filename):
        self.samples = []
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            while True:
                data = f.read(4) # Score (float)
                if not data: break
                score = struct.unpack("f", data)[0]
                
                data = f.read(2) # Count (int16)
                count = struct.unpack("h", data)[0]
                
                data = f.read(2 * count) # Feature IDs
                features = struct.unpack(f"{count}h", data)
                
                feat_vec = torch.zeros(256)
                for fid in features:
                    if 0 <= fid < 256: # Safety check
                        feat_vec[fid] = 1.0
                
                self.samples.append((feat_vec, torch.tensor([score])))
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        # Architecture: 256 -> 256 -> 16 -> 32 -> 1
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

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NNUE().to(device)

full_dataset = NNUEDataset("scripts/training_data.bin")
# Regularization: Use % of data for validation to detect overfitting
train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)

criterion = nn.MSELoss()
# AdamW: Integrated L2 regularization (weight_decay) to keep weights small for int16
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

print(f"Training on {device}...")

best_val_loss = float('inf')

for epoch in range(20): # Increased epochs, early stopping will catch it
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
    print(f"Epoch {epoch+1} | Train Loss: {total_train_loss/len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")

    # Save best model based on validation
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "nnue_model_best.pt")
        print("  --> Model Saved (New Best)")

print("Training finished.")