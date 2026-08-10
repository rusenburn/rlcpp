"""Quantizes and exports a 384-feature layer-stacked NNUE for the C++ engine.

Two things differ from export_nnue_layerstacks.py:

  1. l1.weight is TRANSPOSED before it is written. PyTorch stores a Linear's
     weight as [out_features, in_features], and the old exporter wrote it
     through unchanged, so the C++ accumulator update walked a column of the
     table - a 512-byte stride touching 256 separate cache lines for every
     single feature toggle. Written as [in_features, out_features], one
     toggle is 512 contiguous bytes.

  2. The file starts with a 64-byte header carrying a magic number, the
     version and the layer geometry, so a stale or mismatched weights file is
     rejected on load instead of being read as noise.

Field order after the header must match NNUELayerStacksModelV2 in
nnue/include/nnue/nnue_layerstacks_model_v2.hpp exactly.

Usage:
    python export_nnue_layerstacks_v2.py [in.pt] [out.bin]
"""

import struct
import sys

import numpy as np
import torch

NUM_BUCKETS = 8
NUM_FEATURES = 384
L1_SIZE = 256
L2_SIZE = 16
L3_SIZE = 32

MAGIC = 0x3255594D  # "MYU2" little-endian
VERSION = 2
QUANT_SHIFT = 7     # weights scaled by 1 << 7 == 128, so C++ can use >> 7
S = 1 << QUANT_SHIFT


def _check_range(name, data):
    """Weights are clamped to +/-1.9 during training; anything beyond +/-255
    would overflow the int16 accumulator arithmetic downstream."""
    peak = float(np.abs(data).max()) if data.size else 0.0
    if peak * S > 32767:
        raise ValueError(f"{name}: |w|max={peak:.3f} overflows int16 at scale {S}")
    if peak > 2.0:
        print(f"  note: {name} peaks at {peak:.3f}, above the 1.9 training clamp")


def write_header(f):
    header = struct.pack(
        "<8I32s",
        MAGIC, VERSION, NUM_FEATURES, L1_SIZE, L2_SIZE, L3_SIZE, NUM_BUCKETS, QUANT_SHIFT,
        b"\0" * 32,
    )
    assert len(header) == 64, len(header)
    f.write(header)


def export(pt_file, out_file):
    state_dict = torch.load(pt_file, map_location="cpu")

    def write_weight(f, name, transpose=False):
        data = state_dict[name].numpy()
        if transpose:
            data = np.ascontiguousarray(data.T)
        _check_range(name, data)
        quantized = np.rint(data * S).astype(np.int16)
        f.write(quantized.tobytes())
        print(f"Exported {name}: {quantized.shape} as int16{' (transposed)' if transpose else ''}")

    def write_bias(f, name, hidden_precision=True):
        data = state_dict[name].numpy()
        _check_range(name, data)
        if hidden_precision:
            # L2/L3/output biases live at the (weight * input) magnitude, so
            # they are scaled by S*S and kept as int32.
            quantized = np.rint(data * S * S).astype(np.int32)
            f.write(quantized.tobytes())
            print(f"Exported {name}: {quantized.shape} as int32")
        else:
            # The L1 bias is added straight into the int16 accumulator.
            quantized = np.rint(data * S).astype(np.int16)
            f.write(quantized.tobytes())
            print(f"Exported {name}: {quantized.shape} as int16")

    l1_shape = tuple(state_dict["l1.weight"].shape)
    if l1_shape != (L1_SIZE, NUM_FEATURES):
        raise ValueError(
            f"l1.weight is {l1_shape}, expected ({L1_SIZE}, {NUM_FEATURES}). "
            "This checkpoint is not a 384-feature model."
        )

    with open(out_file, "wb") as f:
        write_header(f)

        # Shared feature transformer, stored [feature][neuron].
        write_weight(f, "l1.weight", transpose=True)
        write_bias(f, "l1.bias", hidden_precision=False)

        # Then each layer's weights for every bucket, then that layer's biases
        # for every bucket, before moving on to the next layer.
        for b in range(NUM_BUCKETS):
            write_weight(f, f"l2.{b}.weight")
        for b in range(NUM_BUCKETS):
            write_bias(f, f"l2.{b}.bias")

        for b in range(NUM_BUCKETS):
            write_weight(f, f"l3.{b}.weight")
        for b in range(NUM_BUCKETS):
            write_bias(f, f"l3.{b}.bias")

        for b in range(NUM_BUCKETS):
            write_weight(f, f"output.{b}.weight")
        for b in range(NUM_BUCKETS):
            write_bias(f, f"output.{b}.bias")

        size = f.tell()

    expected = 64 + (
        NUM_FEATURES * L1_SIZE * 2 + L1_SIZE * 2
        + NUM_BUCKETS * L2_SIZE * L1_SIZE * 2 + NUM_BUCKETS * L2_SIZE * 4
        + NUM_BUCKETS * L3_SIZE * L2_SIZE * 2 + NUM_BUCKETS * L3_SIZE * 4
        + NUM_BUCKETS * L3_SIZE * 2 + NUM_BUCKETS * 4
    )
    if size != expected:
        raise ValueError(
            f"wrote {size} bytes, expected {expected}. The exporter and "
            "NNUELayerStacksModelV2 are out of sync."
        )

    print(f"\nSuccess! {size} bytes written to: {out_file}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "nnue_layerstacks_v2_best.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "nnue_layerstacks_v2_weights.bin"
    export(src, dst)
