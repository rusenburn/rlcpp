cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(This common)
project(${This})
set(SourceFiles 
    src/state.cpp
    src/exceptions.cpp
    src/match.cpp
    src/player.cpp
    src/random.cpp
    )
set(HeaderFiles )
add_library(${This} STATIC ${SourceFiles} ${HeaderFiles})
set_property(TARGET ${This} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
)