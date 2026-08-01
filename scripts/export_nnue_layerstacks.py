import torch
import numpy as np

NUM_BUCKETS = 8


def export_nnue_layerstacks(pt_file, out_file):
    # Load the best model weights
    state_dict = torch.load(pt_file, map_location='cpu')

    # Scaling factor for quantization
    # 128 is perfect because (x * 128) can be shifted right by 7 (>> 7) in C++
    S = 128

    def write_scalar_weight(f, name):
        data = state_dict[name].numpy()
        quantized = (data * S).astype(np.int16)
        f.write(quantized.tobytes())
        print(f"Exported {name}: {quantized.shape} as int16")

    def write_scalar_bias(f, name, hidden_precision=True):
        data = state_dict[name].numpy()
        if hidden_precision:
            # L2/L3/output biases as int32_t, scaled by S*S to match the
            # (Weight * Input) magnitude - same convention as export_nnue.py.
            quantized = (data * S * S).astype(np.int32)
            f.write(quantized.tobytes())
            print(f"Exported {name}: {quantized.shape} as int32")
        else:
            # L1 bias is added directly to the accumulator (int16).
            quantized = (data * S).astype(np.int16)
            f.write(quantized.tobytes())
            print(f"Exported {name}: {quantized.shape} as int16")

    with open(out_file, "wb") as f:
        # Field order must match NNUELayerStacksModel in
        # nnue/include/nnue/nnue_layerstacks_model.hpp exactly: shared L1
        # first, then each layer's weights for ALL buckets, then that
        # layer's biases for all buckets, before moving to the next layer.
        write_scalar_weight(f, 'l1.weight')
        write_scalar_bias(f, 'l1.bias', hidden_precision=False)

        for b in range(NUM_BUCKETS):
            write_scalar_weight(f, f'l2.{b}.weight')
        for b in range(NUM_BUCKETS):
            write_scalar_bias(f, f'l2.{b}.bias')

        for b in range(NUM_BUCKETS):
            write_scalar_weight(f, f'l3.{b}.weight')
        for b in range(NUM_BUCKETS):
            write_scalar_bias(f, f'l3.{b}.bias')

        for b in range(NUM_BUCKETS):
            write_scalar_weight(f, f'output.{b}.weight')
        for b in range(NUM_BUCKETS):
            write_scalar_bias(f, f'output.{b}.bias')

    print(f"\nSuccess! NNUE layer-stacks weights saved to: {out_file}")


if __name__ == "__main__":
    export_nnue_layerstacks("nnue_layerstacks_best.pt", "nnue_layerstacks_weights.bin")
