import torch
import torch.nn as nn

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
    
# Use the same class definition from your training script
model = NNUE() 
model.load_state_dict(torch.load("nnue_model_best.pt"))
model.eval()

# Create dummy input to trace the graph
dummy_input = torch.randn(1, 256)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("nnue_traced.pt")
print("Traced model saved for LibTorch!")