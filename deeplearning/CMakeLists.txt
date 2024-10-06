cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(This deeplearning)
project(${This})

set(CUDNN_LIBRARY_PATH "C:/Program Files/NVIDIA/CUDNN/v8.5.0.96/lib")
set(CUDNN_INCLUDE_PATH "C:/Program Files/NVIDIA/CUDNN/v8.5.0.96/include")
find_package(Torch REQUIRED)

set(SourceFiles 
    src/alphazero/networks/az.cpp
    src/alphazero/networks/resblock.cpp
    src/alphazero/networks/shared_res_nn.cpp
    src/alphazero/networks/smallnn.cpp
    src/alphazero/networks/squeeze_and_excite.cpp
    src/alphazero/networks/tinynn.cpp
    src/alphazero/alphazero.cpp
    src/deeplearning.cpp
    src/network_evaluator.cpp
    )
set(HeaderFiles )

add_library(${This} STATIC ${SourceFiles} ${HeaderFiles})
set_property(TARGET ${This} PROPERTY CXX_STANDARD 17)

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
)


target_link_libraries(${PROJECT_NAME}
    PUBLIC
        common
        players
        games
    PRIVATE
        ${TORCH_LIBRARIES}
    )


set_target_properties(${PROJECT_NAME} PROPERTIES
RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
