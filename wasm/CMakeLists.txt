cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(This gplayer_wasm)
project(${This} VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Emscripten-specific settings
if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".js")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s WASM=1 -s EXPORTED_FUNCTIONS=['_select_move','_malloc','_free'] -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','getValue','setValue','ALLOC_NORMAL'] -s ALLOW_MEMORY_GROWTH=1")
endif()

# Add source files
set(SOURCES
    gplayer_wasm.cpp
)

# Create executable
add_executable(${This} ${SOURCES})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../games/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../players/include
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    common
    players
    games)

# Copy demo files to build directory after build
add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${CMAKE_CURRENT_SOURCE_DIR}/demo.html
                   $<TARGET_FILE_DIR:${PROJECT_NAME}>/demo.html)

add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${CMAKE_CURRENT_SOURCE_DIR}/worker.js
                   $<TARGET_FILE_DIR:${PROJECT_NAME}>/worker.js)
