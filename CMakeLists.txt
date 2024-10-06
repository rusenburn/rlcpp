cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(rlcpp VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/raylib5.0)
add_subdirectory(external/abseil-cpp)
add_subdirectory(common)
add_subdirectory(games)
add_subdirectory(players)
add_subdirectory(deeplearning)
add_subdirectory(run)
add_subdirectory(ui)