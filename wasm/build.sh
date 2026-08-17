#!/bin/bash

# Builds the Migoyugo WebAssembly engine and the static site around it.
# Run from the wasm/ directory with Emscripten activated:
#
#   source /path/to/emsdk/emsdk_env.sh
#   ./build.sh

set -e

if ! command -v emcmake &> /dev/null; then
    echo "Error: emcmake not found. Activate Emscripten first:"
    echo "  source /path/to/emsdk/emsdk_env.sh"
    echo "  cd /path/to/rlcpp/wasm && ./build.sh"
    exit 1
fi

cd ..

if [ ! -f checkpoints/nnue_layerstacks_v2_weights.bin ]; then
    echo "Error: checkpoints/nnue_layerstacks_v2_weights.bin is missing."
    echo "The network is a runtime asset and is deliberately untracked. Export it:"
    echo "  cd scripts && python export_nnue_layerstacks_v2.py \\"
    echo "      nnue_layerstacks_v2_best.pt ../checkpoints/nnue_layerstacks_v2_weights.bin"
    exit 1
fi

# The AlphaZero network is optional: it is 40 MB and only az.html needs it, so a
# missing one is a warning rather than a hard stop the way the NNUE weights are.
if [ ! -f checkpoints/migoyugo_az.onnx ]; then
    echo "Note: checkpoints/migoyugo_az.onnx is missing, so az.html will 404 on"
    echo "its network. The NNUE engine at index.html is unaffected. Export it:"
    echo "  cd scripts && python export_az_onnx.py \\"
    echo "      ../checkpoints/migoyugo_strongest_900.pt ../checkpoints/migoyugo_az.onnx"
    echo ""
fi

emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release
# Default target: both engines and both sets of web assets.
cmake --build build/wasm -j"$(nproc 2>/dev/null || echo 4)"

echo ""
echo "Build complete. The whole site is in build/wasm/web/:"
echo "  NNUE alpha-beta engine (index.html):"
echo "    index.html  styles.css  app.js  board.js  sound.js  snapshot.js  worker.js"
echo "    migoyugo.js  migoyugo.wasm  migoyugo_nnue_v2.bin"
echo "  AlphaZero bot on WebGPU (az.html):"
echo "    az.html  az_worker.js  az_snapshot.js"
echo "    migoyugo_az.js  migoyugo_az.wasm  migoyugo_az.onnx"
echo ""
echo "Serve it (WebAssembly and fetch() both need http://, not file://):"
echo "  python3 -m http.server 8000 --directory build/wasm/web"
echo "  then open http://localhost:8000/        (NNUE engine)"
echo "            http://localhost:8000/az.html (AlphaZero bot)"
echo ""
echo "AlphaYugo needs a WebGPU-capable browser for the GPU path; without one it"
echo "falls back to onnxruntime-web's CPU backend and reports that in the panel."
echo ""
echo "On Linux with hybrid graphics, all four of these are load-bearing:"
echo ""
echo "  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \\"
echo "  google-chrome --user-data-dir=/tmp/chrome-webgpu \\"
echo "    --enable-unsafe-webgpu --enable-features=Vulkan,VulkanFromANGLE \\"
echo "    --use-angle=vulkan http://localhost:8000/"
echo ""
echo "  VK_DRIVER_FILES     the Vulkan loader otherwise enumerates the integrated"
echo "                      GPU first; on Ivy Bridge that driver is incomplete and"
echo "                      the discrete card is never reached."
echo "  --use-angle=vulkan  and --enable-features=Vulkan put Chrome's GPU stack on"
echo "                      real Vulkan. WITHOUT THEM an adapter is still handed"
echo "                      out and the page still reports 'webgpu', but it is a"
echo "                      software one: measured 195 ms per batch of 16 against"
echo "                      16 ms on the real GPU. Do not drop them just because"
echo "                      requestAdapter() succeeds - that only proves some"
echo "                      adapter exists, not that it is the fast one."
echo "  --user-data-dir     flags are ignored if Chrome is already running; a"
echo "                      separate profile forces a fresh process."
echo ""

cd wasm
