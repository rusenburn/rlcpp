import torch
import numpy as np
import struct

def export_nnue(pt_file, out_file):
    # Load the best model weights
    state_dict = torch.load(pt_file, map_location='cpu')
    
    # Scaling factor for quantization
    # 128 is perfect because (x * 128) can be shifted right by 7 (>> 7) in C++
    S = 128 

    with open(out_file, "wb") as f:
        # Layer order must match C++ loading order
        layers = [
            'l1.weight', 'l1.bias', 
            'l2.weight', 'l2.bias', 
            'l3.weight', 'l3.bias', 
            'output.weight', 'output.bias'
        ]

        for name in layers:
            data = state_dict[name].numpy()
            
            if 'weight' in name:
                # Weights are int16_t
                # Transpose weight matrices so they are [Output][Input] for easier C++ loops
                quantized = (data * S).astype(np.int16)
                f.write(quantized.tobytes())
                print(f"Exported {name}: {quantized.shape} as int16")
            
            else:
                # Biases for hidden layers often benefit from higher precision
                # We save hidden biases as int32_t to prevent rounding drift
                if name == 'l1.bias':
                     # L1 bias is added directly to the accumulator (int16)
                     quantized = (data * S).astype(np.int16)
                     f.write(quantized.tobytes())
                     print(f"Exported {name}: {quantized.shape} as int16")
                else:
                     # L2, L3, and Output biases as int32_t
                     # We scale them by S*S (16384) to match the (Weight * Input) magnitude
                     quantized = (data * S * S).astype(np.int32)
                     f.write(quantized.tobytes())
                     print(f"Exported {name}: {quantized.shape} as int32")

    print(f"\nSuccess! NNUE weights saved to: {out_file}")

if __name__ == "__main__":
    export_nnue("nnue_model_best.pt", "nnue_weights.bin")