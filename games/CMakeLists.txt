cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(This games)
project(${This})
set(SourceFiles 
    src/damma.cpp
    src/english_draughts.cpp
    src/gobblet_goblers.cpp
    src/migoyugo.cpp
    src/tictactoe.cpp
    src/othello.cpp
    src/santorini.cpp
    src/walls.cpp
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

target_link_libraries(${PROJECT_NAME} common)
