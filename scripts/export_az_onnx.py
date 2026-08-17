"""Exports the C++ AlphaZero residual network (SharedResNetwork) to ONNX.

libtorch's C++ API has no ONNX exporter - torch.onnx.export is Python-only, and
the nets here are torch::nn::Module, not TorchScript. What makes this script
possible is that a C++ `torch::save(mod_, path)` archive loads in Python through
torch.jit.load(), and the parameter names it yields are exactly the names the
C++ side passes to register_module():

    shared_.0.weight            shared_.1.block_.0.weight
    shared_.1.se_.fcs_.1.weight probs_head_.4.weight  wdls_head_.4.weight

So the module below is a mirror of SharedResImpl with *identical submodule
names*, and load_state_dict(strict=True) is what proves the mirror is faithful:
a single renamed or reshaped layer fails the load rather than silently exporting
a differently-wired net.

Two deliberate choices:

  1. forward() keeps the `x - x.logsumexp(-1, True)` that
     shared_res_nn.cpp does before each softmax. Softmax is shift-invariant, so
     it is mathematically a no-op, but reproducing it keeps this a faithful
     export of the graph rather than a reinterpretation of it.

  2. The action mask and the wdl -> scalar reduction stay OUT of the graph.
     NetworkEvaluator applies them after forward(), and keeping the split in the
     same place lets the C++ OnnxEvaluator be diffed against NetworkEvaluator
     step for step. It also keeps the .onnx reusable by anything that wants raw
     policy and wdl.

Geometry is inferred from the checkpoint's tensor shapes, so this works on any
SharedResNetwork checkpoint rather than only the 8x8 Migoyugo ones.

Usage:
    python export_az_onnx.py [in.pt] [out.onnx]
"""

import argparse
import sys

import torch
import torch.nn as nn

OPSET = 17


class SqueezeAndExcite(nn.Module):
    """Mirror of SqueezeAndExciteImpl (squeeze_and_excite.cpp).

    Sequential indices matter: prepare_ is a one-element Sequential so the pool
    is prepare_.0, and fcs_ is Flatten/Linear/ReLU/Linear so the weights land on
    fcs_.1 and fcs_.3 - which is what the checkpoint keys say.
    """

    def __init__(self, channels, squeeze_rate=4):
        super().__init__()
        self.channels_ = channels
        self.prepare_ = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.fcs_ = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // squeeze_rate),
            nn.ReLU(),
            nn.Linear(channels // squeeze_rate, channels * 2),
        )

    def forward(self, state, input_):
        prepared = self.fcs_(self.prepare_(state))
        w, b = prepared.split(self.channels_, 1)
        z = torch.sigmoid(w).unsqueeze(-1).unsqueeze(-1).expand_as(input_)
        b = b.unsqueeze(-1).unsqueeze(-1).expand_as(input_)
        return state * z + b


class ResBlockSE(nn.Module):
    """Mirror of ResBlockSEImpl (resblock.cpp).

    Note there is no activation between the two convolutions. That is what the
    C++ does; do not "fix" it here or the export stops matching the weights.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.block_ = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1),
            nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1),
        )
        self.se_ = SqueezeAndExcite(n_channels, 4)

    def forward(self, state):
        out = self.block_(state)
        out = self.se_(out, state)
        return torch.relu(out + state)


class SharedRes(nn.Module):
    """Mirror of SharedResImpl (shared_res_nn.cpp)."""

    def __init__(self, observation_shape, n_actions, filters, fc_dims, n_blocks,
                 normalize_outputs):
        super().__init__()
        c, h, w = observation_shape
        self.normalize_outputs_ = normalize_outputs

        blocks = [nn.Conv2d(c, filters, 3, stride=1, padding=1)]
        blocks += [ResBlockSE(filters) for _ in range(n_blocks)]
        self.shared_ = nn.Sequential(*blocks)

        def head(out_dims):
            return nn.Sequential(
                nn.Conv2d(filters, filters, 3, stride=1, padding=1),
                nn.Flatten(),
                nn.Linear(h * w * filters, fc_dims),
                nn.ReLU(),
                nn.Linear(fc_dims, out_dims),
            )

        self.probs_head_ = head(n_actions)
        self.wdls_head_ = head(3)

    def forward(self, state):
        shared = self.shared_(state)
        probs = self.probs_head_(shared)
        wdls = self.wdls_head_(shared)
        if self.normalize_outputs_:
            probs = probs - probs.logsumexp(-1, True)
            wdls = wdls - wdls.logsumexp(-1, True)
        return probs.softmax(-1), wdls.softmax(-1)


def load_cpp_state_dict(pt_path):
    """Reads a checkpoint written by libtorch's torch::save(module, path).

    These are serialization archives, not TorchScript modules, but torch.jit.load
    accepts them and exposes the tensors as a state_dict. torch.load does not.
    """
    try:
        container = torch.jit.load(pt_path, map_location="cpu")
    except Exception as exc:
        raise SystemExit(
            f"{pt_path}: could not be read as a libtorch archive ({exc}).\n"
            "Expected a file written by SharedResNetwork::save() / torch::save()."
        )
    return {k: v.detach().cpu() for k, v in container.state_dict().items()}


def infer_geometry(sd, height=None, width=None):
    """Recovers the constructor arguments from the tensor shapes.

    SharedResNetwork::save() stores no hyperparameters (only save_full does), so
    every C++ caller hardcodes `128, 512, 5, true`. Deriving them instead means a
    checkpoint trained with different settings exports correctly rather than
    silently failing the strict load.
    """
    try:
        stem = sd["shared_.0.weight"]                # [filters, C, 3, 3]
        fc1 = sd["probs_head_.2.weight"]             # [fc_dims, H*W*filters]
        out = sd["probs_head_.4.weight"]             # [n_actions, fc_dims]
    except KeyError as exc:
        raise SystemExit(
            f"checkpoint is missing {exc}; this does not look like a "
            "SharedResNetwork (SmallAlpha and Tiny nets are not supported)."
        )

    filters, channels = stem.shape[0], stem.shape[1]
    fc_dims = fc1.shape[0]
    n_actions = out.shape[0]
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("shared_.")}) - 1

    plane = fc1.shape[1] // filters
    if height is None and width is None:
        side = int(round(plane ** 0.5))
        if side * side != plane:
            raise SystemExit(
                f"board has {plane} cells, which is not square; pass --height/--width."
            )
        height = width = side
    elif height is None:
        height = plane // width
    elif width is None:
        width = plane // height
    if height * width != plane:
        raise SystemExit(f"--height {height} x --width {width} != {plane} cells")

    return (channels, height, width), n_actions, filters, fc_dims, n_blocks


def verify(model, onnx_path, observation_shape, atol):
    """Runs torch and onnxruntime on the same input and compares.

    Two batch sizes on purpose: batch 1 is what the evaluator sees for a single
    state, and batch 8 proves the dynamic axis actually stayed dynamic instead of
    being frozen at the export batch size.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed - skipping verification", file=sys.stderr)
        return True

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ok = True

    for batch in (1, 8):
        x = torch.randn(batch, *observation_shape)
        with torch.no_grad():
            ref_probs, ref_wdl = model(x)
        got_probs, got_wdl = session.run(None, {input_name: x.numpy()})

        d_probs = (ref_probs - torch.from_numpy(got_probs)).abs().max().item()
        d_wdl = (ref_wdl - torch.from_numpy(got_wdl)).abs().max().item()
        status = "ok" if max(d_probs, d_wdl) < atol else "FAILED"
        if status == "FAILED":
            ok = False
        print(f"  batch {batch:>2}: probs {d_probs:.3e}  wdl {d_wdl:.3e}  {status}")

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default="../checkpoints/migoyugo_strongest_900.pt")
    parser.add_argument("output", nargs="?", default="../checkpoints/migoyugo_az.onnx")
    parser.add_argument("--height", type=int, default=None,
                        help="board rows; only needed for non-square boards")
    parser.add_argument("--width", type=int, default=None,
                        help="board cols; only needed for non-square boards")
    parser.add_argument("--no-normalize-outputs", action="store_true",
                        help="mirror a net built with normalize_outputs=false")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="tolerance for the torch-vs-onnxruntime check")
    args = parser.parse_args()

    print(f"Reading {args.input}")
    sd = load_cpp_state_dict(args.input)
    shape, n_actions, filters, fc_dims, n_blocks = infer_geometry(sd, args.height, args.width)
    print(f"  observation {shape[0]}x{shape[1]}x{shape[2]}, {n_actions} actions, "
          f"{filters} filters, {fc_dims} fc dims, {n_blocks} blocks, "
          f"{sum(t.numel() for t in sd.values()):,} parameters")

    model = SharedRes(shape, n_actions, filters, fc_dims, n_blocks,
                      not args.no_normalize_outputs)
    model.load_state_dict(sd, strict=True)   # the real correctness gate
    model.eval()

    print(f"Exporting to {args.output} (opset {OPSET})")
    torch.onnx.export(
        model,
        torch.zeros(1, *shape),
        args.output,
        input_names=["observation"],
        output_names=["probs", "wdl"],
        dynamic_axes={"observation": {0: "batch"},
                      "probs": {0: "batch"},
                      "wdl": {0: "batch"}},
        opset_version=OPSET,
        # torch 2.10 defaults to dynamo=True, which needs onnxscript and emits
        # graphs with weaker onnxruntime-web coverage. The tracing exporter is
        # the portable choice for a static graph like this one.
        dynamo=False,
    )

    print("Verifying against onnxruntime")
    if not verify(model, args.output, shape, args.atol):
        raise SystemExit("verification failed: the exported graph does not match torch")
    print("Done.")


if __name__ == "__main__":
    main()
