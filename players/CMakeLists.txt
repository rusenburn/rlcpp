cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(This players)
project(${This})

# AMCTS2 and the evaluator interface build everywhere, including WASM: the
# browser's ONNX bot is an Amcts2 driven by an OnnxEvaluator.
#
# These used to sit in the NOT EMSCRIPTEN list below, described as having
# "structured binding issues". They do not - compiling all four under emcc 6.0.6
# produces warnings only, no errors. Whatever the original breakage was, it is
# gone.
#
# Note this is Amcts2 itself, not Amcts2Player: amcts2_player.cpp also defines
# ConcurrentPlayer, which drags in ConcurrentAmcts and std::thread and would make
# the WASM link need -pthread for no benefit. The wasm shim constructs Amcts2
# directly, the way run/onnx_parity.cpp does.
set(PORTABLE_SEARCH_SOURCES
    src/evaluator.cpp
    src/bandits/amcts2/amcts2.cpp
    src/bandits/amcts2/amcts2_node.cpp
    )

# Still native-only. Everything here needs threads or has not been tried under
# emcc; none of it is on the browser path.
if(NOT EMSCRIPTEN)
    set(EXCLUDED_SOURCES
        src/bandits/amcts2/concurrent_amcts.cpp
        src/amcts_player.cpp
        src/amcts.cpp
        src/amcts2_player.cpp
        src/concurrent_search_tree.cpp
        src/evaluator_player.cpp
        # LM-MCTS sources (reference binding issues)
        src/bandits/lm_mcts/lm_mcts_node.cpp
        src/bandits/lm_mcts/lm_mcts.cpp
        src/lm_mcts_player.cpp
    )
endif()

set(SourceFiles
    ${PORTABLE_SEARCH_SOURCES}
    ${EXCLUDED_SOURCES}
    src/bandits/grave/g_node.cpp
    src/bandits/grave/g.cpp
    src/bandits/grave/grave_node.cpp
    src/bandits/grave/grave.cpp
    src/bandits/mcrave/mcrave.cpp
    src/bandits/mcrave/mcrave_node.cpp
    src/bandits/uct/uct_node.cpp
    src/bandits/uct/uct.cpp
    src/grave_player.cpp
    src/g_player.cpp
    src/human_player.cpp
    src/mcrave_player.cpp
    src/mcts_player.cpp
    src/mcts.cpp
    src/random_action_player.cpp
    src/random_evaluator.cpp
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
