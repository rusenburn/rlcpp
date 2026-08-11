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

emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm --target migoyugo_wasm -j"$(nproc 2>/dev/null || echo 4)"

echo ""
echo "Build complete. The whole site is in build/wasm/web/:"
echo "  index.html  styles.css  app.js  board.js  sound.js  snapshot.js  worker.js"
echo "  migoyugo.js  migoyugo.wasm  migoyugo_nnue_v2.bin"
echo ""
echo "Serve it (WebAssembly and fetch() both need http://, not file://):"
echo "  python3 -m http.server 8000 --directory build/wasm/web"
echo "  then open http://localhost:8000/"
echo ""

cd wasm
