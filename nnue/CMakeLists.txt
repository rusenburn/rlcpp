cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(This nnue)
project(${This})

# Exclude problematic sources for WASM builds (structured binding and reference issues)
if(NOT EMSCRIPTEN)
    set(EXCLUDED_SOURCES
    )
endif()

set(SourceFiles
    ${EXCLUDED_SOURCES}
    src/nnue_mcts_player.cpp
    src/nnue_player.cpp
    src/nnue_mcts.cpp
    )

set(HeaderFiles 
    )

add_library(${This} STATIC ${SourceFiles} ${HeaderFiles})
set_property(TARGET ${This} PROPERTY CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
)

target_link_libraries(${PROJECT_NAME} common)
target_link_libraries(${PROJECT_NAME} games)
