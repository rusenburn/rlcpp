cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(This common)
project(${This})
set(SourceFiles 
    src/concurrent_match.cpp
    src/concurrent_player.cpp
    src/exceptions.cpp
    src/match.cpp
    src/observer.cpp
    src/player.cpp
    src/random.cpp
    src/round_robin.cpp
    src/state.cpp
    src/utils.cpp
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