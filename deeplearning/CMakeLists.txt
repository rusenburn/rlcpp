cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(This deeplearning)
project(${This})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_PREFIX_PATH "C:/libtorch-win-shared-with-deps-1.12.1+cu116/libtorch")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
message("###  Using Debug ###")
# set(CMAKE_PREFIX_PATH "C:/libtorch-win-shared-with-deps-debug-1.12.1+cu116/libtorch")
set(CMAKE_PREFIX_PATH "C:/libtorch-win-shared-with-deps-debug-2.2.1+cu118/libtorch")
message("### CMAKE_PREFIX PATH ${CMAKE_PREFIX_PATH} ###")
endif()

set(CUDNN_LIBRARY_PATH "C:/Program Files/NVIDIA/CUDNN/v8.5.0.96/lib")
set(CUDNN_INCLUDE_PATH "C:/Program Files/NVIDIA/CUDNN/v8.5.0.96/include")
find_package(Torch REQUIRED)

set(SourceFiles 
    src/alphazero/networks/az.cpp
    src/alphazero/networks/resblock.cpp
    src/alphazero/networks/shared_res_nn.cpp
    src/alphazero/networks/smallnn.cpp
    src/alphazero/networks/squeeze_and_excite.cpp
    src/alphazero/alphazero_sub_tree.cpp
    src/alphazero/alphazero_sub_tree2.cpp
    src/alphazero/alphazero.cpp
    src/network_evaluator.cpp
    )
set(HeaderFiles )

# set(CMAKE CXX_STANDARD 17)
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

# # The following code block is suggested to be used on Windows.
# # According to https://github.com/pytorch/pytorch/issues/25457,
# # the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET ${PROJECT_NAME}
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:${PROJECT_NAME}>)
                     
# endif (MSVC)