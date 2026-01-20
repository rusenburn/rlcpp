#!/bin/bash

# Build script for GPlayer WASM module (Unix/Linux/macOS)
# Run this from the wasm/ directory
# Requires Emscripten to be activated

set -e

echo "Building GPlayer WASM module..."

# Check if Emscripten is available
if ! command -v emcmake &> /dev/null; then
    echo "Error: emcmake not found. Please activate Emscripten environment:"
    echo "  source /path/to/emsdk/emsdk_env.sh"
    echo "  cd /path/to/rlcpp/wasm"
    echo "  ./build.sh"
    exit 1
fi

# Move to project root to access other projects
cd ..

# Create build directory structure if needed
mkdir -p build/build_wasm_full

# Configure and build using CMake source (-S) and binary (-B) directory flags
emcmake cmake -S . -B build/build_wasm_full -DCMAKE_BUILD_TYPE=Release
cmake --build build/build_wasm_full --config Release --target gplayer_wasm

echo ""
echo "Build complete!"
echo "All files created in build/build_wasm_full/ directory:"
echo "  - gplayer_wasm.js (JavaScript glue code)"
echo "  - gplayer_wasm.wasm (WebAssembly binary)"
echo "  - demo.html (Demo web page)"
echo "  - worker.js (Web Worker)"
echo ""
echo "To test in browser:"
echo "1. Start a web server: python3 -m http.server 8000"
echo "2. Open http://localhost:8000/build/build_wasm_full/wasm/demo.html"
echo ""

cd wasm
