cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(This run)
project(${This})
set(SourceFiles 
    main.cpp
    console.cpp
    main_console.cpp
    train_ai_console.cpp
    match_console.cpp
    concurrent_match_console.cpp
    )

set(HeaderFiles )

set(CUDNN_LIBRARY_PATH "C:/Program Files/NVIDIA/CUDNN/v8.5.0.96/lib")
set(CUDNN_INCLUDE_PATH "C:/Program Files/NVIDIA/CUDNN/v8.5.0.96/include")
find_package(Torch REQUIRED)

add_executable(${This} ${SourceFiles} ${HeaderFiles})
set_property(TARGET ${This} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    deeplearning 
    "${TORCH_LIBRARIES}")

# by chatgpt
set_target_properties(${PROJECT_NAME} PROPERTIES
RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)

# by chatgpt
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env "PATH=$<TARGET_FILE_DIR:${PROJECT_NAME}>;$ENV{PATH}" $<TARGET_FILE:${PROJECT_NAME}>
    )
endif()


# train_az.exe
project(train_az)
add_executable(train_az train_az.cpp)
set_property(TARGET train_az PROPERTY CXX_STANDARD 17)
set_property(TARGET train_az PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(train_az
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
)
target_link_libraries(train_az PUBLIC
    deeplearning 
    "${TORCH_LIBRARIES}"
    absl::algorithm
    absl::base
    absl::debugging
    absl::flat_hash_map
    absl::flags
    absl::flags_parse
    absl::log
    absl::memory
    absl::meta
    absl::numeric
    absl::random_random
    absl::strings
    absl::synchronization
    absl::time
    absl::utility
    )

set_target_properties(train_az PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET train_az
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:train_az>)
endif (MSVC)

# by chatgpt
if(WIN32)
    add_custom_command(TARGET train_az POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env "PATH=$<TARGET_FILE_DIR:train_az;$ENV{PATH}" $<TARGET_FILE:train_az>
    )
endif()

# convert
set(This conv)
project(${This})
set(SourceFiles 
    convert_to_full_module.cpp
    )

set(HeaderFiles )

add_executable(${This} ${SourceFiles} ${HeaderFiles})
set_property(TARGET ${This} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    deeplearning 
    "${TORCH_LIBRARIES}")

# by chatgpt
set_target_properties(${PROJECT_NAME} PROPERTIES
RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)

# by chatgpt
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env "PATH=$<TARGET_FILE_DIR:${PROJECT_NAME}>;$ENV{PATH}" $<TARGET_FILE:${PROJECT_NAME}>
    )
endif()