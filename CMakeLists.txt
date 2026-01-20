cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(rlcpp VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Only add raylib for non-WASM builds (has GUI dependencies that don't work in WASM)
if(NOT EMSCRIPTEN)
    add_subdirectory(external/raylib5.0)
endif()

add_subdirectory(external/abseil-cpp)
add_subdirectory(common)
add_subdirectory(games)
add_subdirectory(players)

# Only add components that require deep learning for non-WASM builds
if(NOT EMSCRIPTEN)
    add_subdirectory(deeplearning)
    add_subdirectory(run)
    add_subdirectory(ui)
endif()

add_subdirectory(wasm)
