cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(This players)
project(${This})

set(SourceFiles 
    src/bandits/grave/g_node.cpp
    src/bandits/grave/g.cpp
    src/bandits/grave/grave_node.cpp
    src/bandits/grave/grave.cpp
    src/bandits/mcrave/mcrave.cpp
    src/bandits/mcrave/mcrave_node.cpp
    src/bandits/uct/uct_node.cpp
    src/bandits/uct/uct.cpp
    src/amcts_player.cpp
    src/amcts.cpp
    src/evaluator_player.cpp
    src/evaluator.cpp
    src/grave_player.cpp
    src/g_player.cpp
    src/human_player.cpp
    src/mcrave_player.cpp
    src/mcts_player.cpp
    src/mcts.cpp
    src/random_action_player.cpp
    src/random_rollout_evaluator.cpp
    src/search_tree.cpp
    src/uct_player.cpp
    )

set(HeaderFiles 
    include/players/players.hpp
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