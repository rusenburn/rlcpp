@echo off
REM Build script for GPlayer WASM module (Windows)
REM Run this from the wasm/ directory
REM Requires Emscripten to be activated

echo Building GPlayer WASM module...

REM Check if Emscripten is available
where emcmake >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: emcmake not found. Please activate Emscripten environment:
    echo   cd ..\emsdk
    echo   emsdk_env.bat
    echo   cd ..\rlcpp\wasm
    echo   build.bat
    exit /b 1
)

REM Move to project root to access other projects
cd ..

REM Create build directory structure if needed
if not exist build\build_wasm_full mkdir build\build_wasm_full

REM Copy demo files to build directory
@REM copy wasm\demo.html build\build_wasm_full\
@REM copy wasm\worker.js build\build_wasm_full\

REM Configure and build using CMake source (-S) and binary (-B) directory flags
call emcmake cmake -S . -B build/build_wasm_full -DCMAKE_BUILD_TYPE=Release
cmake --build build/build_wasm_full --config Release --target gplayer_wasm

echo.
echo Build complete!
echo All files created in build/build_wasm_full/ directory:
echo   - gplayer_wasm.js (JavaScript glue code)
echo   - gplayer_wasm.wasm (WebAssembly binary)
echo   - demo.html (Demo web page)
echo   - worker.js (Web Worker)
echo.
echo To test in browser:
echo 1. Start a web server: python -m http.server 8000
echo 2. Open http://localhost:8000/build/build_wasm_full/demo.html
echo.

cd wasm
