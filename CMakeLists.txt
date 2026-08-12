cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(rlcpp VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The NNUE evaluation and the Migoyugo bitboards use SSE2/SSE4 intrinsics and
# popcount directly, so the architecture baseline has to be stated rather than
# left to whatever the default target happens to be. Ivy Bridge (i7-3770) has
# SSE4.2, AVX and POPCNT but no AVX2 and no BMI, so 128-bit integer SIMD is the
# ceiling; -mavx buys the non-destructive VEX encodings of the same
# instructions, which keeps register copies out of the accumulator loops.
#
# Until now these flags existed only as a hand-edited CMAKE_CXX_FLAGS in
# build/Release/CMakeCache.txt, so a fresh configure silently produced a much
# slower binary.
if(NOT EMSCRIPTEN)
    if(MSVC)
        add_compile_options(/arch:AVX)
    else()
        add_compile_options(-mavx -msse4.2 -mpopcnt)
        add_compile_options($<$<CONFIG:Release>:-O3>)
        add_compile_options($<$<CONFIG:RelWithDebInfo>:-O2>)
    endif()
endif()

# WebAssembly baseline. EMSCRIPTEN is only ever defined by the toolchain file
# emcmake injects, so nothing here can reach a native build.
#
# These are directory properties and must be set BEFORE the add_subdirectory
# calls below: add_subdirectory snapshots the parent's properties at the point
# of the call, and the whole engine is header-inlined into nnue/, so putting
# the flags on the wasm target alone would leave the accumulator and the
# evaluation compiled as scalar code.
#
#  -msimd128  enables the WebAssembly SIMD proposal. Every intrinsic the engine
#             uses lowers to a single wasm instruction: _mm_add/sub_epi16 ->
#             i16x8.add/sub, _mm_madd_epi16 -> i32x4.dot_i16x8_s,
#             _mm_min/max_epi16 -> i16x8.min/max_s, _mm_load_si128 -> v128.load.
#  -msse2     makes emscripten's <emmintrin.h> shim visible; the shim is gated
#             on __SSE2__, so -msimd128 on its own still fails to compile
#             nnue_layerstacks_player_v2.hpp's <immintrin.h>.
#  -O3        stated explicitly because Emscripten.cmake overrides
#             CMAKE_CXX_FLAGS_RELEASE to "-DNDEBUG -O2" - CMAKE_BUILD_TYPE=Release
#             alone does NOT give -O3 here. Directory COMPILE_OPTIONS are
#             appended after the per-config flags, so the later -O3 wins.
if(EMSCRIPTEN)
    add_compile_options(-msimd128 -msse2 -O3)
    add_link_options(-O3)
endif()

# Only add raylib for non-WASM builds (has GUI dependencies that don't work in WASM)
# Abseil is in the same boat: only run/ links it, and run/ is excluded below,
# so a WASM build was spending minutes compiling it for nothing.
if(NOT EMSCRIPTEN)
    add_subdirectory(external/raylib5.0)
    add_subdirectory(external/abseil-cpp)
endif()

add_subdirectory(common)
add_subdirectory(games)
add_subdirectory(players)
add_subdirectory(nnue)

# Only add components that require deep learning for non-WASM builds
if(NOT EMSCRIPTEN)
    add_subdirectory(deeplearning)
    add_subdirectory(run)
    add_subdirectory(ui)
endif()

if (EMSCRIPTEN)
    add_subdirectory(wasm)
endif()