@echo off
REM Builds the Migoyugo WebAssembly engine and the static site around it.
REM Run from the wasm\ directory with Emscripten activated.

where emcmake >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: emcmake not found. Activate Emscripten first:
    echo   cd ..\emsdk ^&^& emsdk_env.bat
    echo   cd ..\rlcpp\wasm ^&^& build.bat
    exit /b 1
)

cd ..

if not exist checkpoints\nnue_layerstacks_v2_weights.bin (
    echo Error: checkpoints\nnue_layerstacks_v2_weights.bin is missing.
    echo The network is a runtime asset and is deliberately untracked. Export it:
    echo   cd scripts ^&^& python export_nnue_layerstacks_v2.py ^
    echo       nnue_layerstacks_v2_best.pt ..\checkpoints\nnue_layerstacks_v2_weights.bin
    cd wasm
    exit /b 1
)

call emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm --target migoyugo_wasm

echo.
echo Build complete. The whole site is in build\wasm\web\:
echo   index.html  styles.css  app.js  board.js  sound.js  snapshot.js  worker.js
echo   migoyugo.js  migoyugo.wasm  migoyugo_nnue_v2.bin
echo.
echo Serve it (WebAssembly and fetch() both need http://, not file://):
echo   python -m http.server 8000 --directory build/wasm/web
echo   then open http://localhost:8000/
echo.

cd wasm
