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

# Only add raylib for non-WASM builds (has GUI dependencies that don't work in WASM)
if(NOT EMSCRIPTEN)
    add_subdirectory(external/raylib5.0)
endif()

add_subdirectory(external/abseil-cpp)
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