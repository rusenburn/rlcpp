cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(This migoyugo_wasm)
project(${This} VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "wasm/ is only built under emcmake; see wasm/build.sh")
endif()

set(CMAKE_EXECUTABLE_SUFFIX ".js")

add_executable(${This} migoyugo_wasm.cpp)

target_link_libraries(${This} PRIVATE nnue games common)

# The page loads this as migoyugo.js (see web/worker.js), not as the target name.
set_target_properties(${This} PROPERTIES OUTPUT_NAME migoyugo)

# The C functions the browser calls. _malloc/_free are needed so the worker can
# hand the fetched weights into linear memory.
#
# One quoted string on purpose: written as separate CMake arguments this
# becomes a list, and the semicolons CMake joins it with end up inside the
# symbol names ("undefined exported symbol: ;_mgy_bot_move").
set(MGY_EXPORTS "_malloc,_free,_mgy_init,_mgy_new_game,_mgy_play,_mgy_undo,_mgy_load_moves,_mgy_bot_move,_mgy_bot_suggest,_mgy_set_time_ms,_mgy_set_tt_mb,_mgy_clear_tt,_mgy_snapshot,_mgy_snapshot_size,_mgy_info,_mgy_info_size")

target_link_options(${This} PRIVATE
    # A factory function instead of a global Module, so worker.js never has to
    # assign to an implicit global and never sees a half-initialised module.
    -sMODULARIZE=1
    -sEXPORT_NAME=createMigoyugo
    # The module only ever runs inside the Web Worker.
    -sENVIRONMENT=worker

    # STACK_SIZE is not tuning: emscripten's default is 64 KB, and a search()
    # frame is about 1 KB (MoveList 516 B + FeatureDelta 328 B + Undo + spills)
    # recursing until the ply >= MAX_PLY - 2 guard, so roughly 90 KB worst
    # case. Under -O3 -DNDEBUG a wasm stack overflow silently overwrites static
    # data, which would look like the engine playing garbage rather than like a
    # crash. 4 MB is linear address space and costs nothing.
    -sSTACK_SIZE=4MB
    # Sized so a default game (16 MB table + 273 KB model + stack) never has to
    # grow; growth still works if the user asks for a bigger table.
    -sINITIAL_MEMORY=48MB
    -sALLOW_MEMORY_GROWTH=1
    -sMALLOC=emmalloc

    # The weights arrive as bytes from fetch(), so nothing needs a filesystem.
    -sFILESYSTEM=0
    -sEXIT_RUNTIME=0
    -sINVOKE_RUN=0

    # No exception flags on purpose. Emscripten's default turns a throw into an
    # abort that kills the module permanently, so migoyugo_wasm.cpp is written
    # to be throw-free rather than paying for -fwasm-exceptions.
    "-sEXPORTED_FUNCTIONS=[${MGY_EXPORTS}]"
    -sEXPORTED_RUNTIME_METHODS=[HEAPU8,HEAPU32,HEAPF64]
)

set_target_properties(${This} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/web)

# CMake creates the output directory at generate time, so deleting build/wasm/web
# to force a fresh site leaves the next link failing with "output file is in a
# directory that does not exist" until you re-run cmake. Recreate it here so
# that is just a rebuild.
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/web)

# --- static site assets ------------------------------------------------------
#
# Everything the browser needs ends up in one directory, so the build output is
# the thing you serve or deploy.

set(MGY_WEB_FILES
    web/index.html
    web/styles.css
    web/app.js
    web/board.js
    web/sound.js
    web/snapshot.js
    web/worker.js)

# A target that always runs, rather than POST_BUILD commands on the engine.
# POST_BUILD only fires when the target relinks, so editing only a .css or .js
# would leave the build output stale - which looks like your change having no
# effect. copy_if_different makes running every build free.
set(MGY_COPY_COMMANDS "")
foreach(f ${MGY_WEB_FILES})
    get_filename_component(name ${f} NAME)
    list(APPEND MGY_COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/${f}
        ${CMAKE_BINARY_DIR}/web/${name})
endforeach()

add_custom_target(migoyugo_web ALL
    ${MGY_COPY_COMMANDS}
    COMMENT "Copying web assets to ${CMAKE_BINARY_DIR}/web"
    VERBATIM)
add_dependencies(migoyugo_web ${This})

# The network weights are a runtime asset, not a build input. They are
# deliberately untracked (see .gitignore), so a missing file must fail the
# configure loudly rather than produce a site that 404s on its own brain.
set(MGY_WEIGHTS ${CMAKE_SOURCE_DIR}/checkpoints/nnue_layerstacks_v2_weights.bin)
if(NOT EXISTS ${MGY_WEIGHTS})
    message(FATAL_ERROR
        "Missing ${MGY_WEIGHTS}\n"
        "Export it first:\n"
        "  cd scripts && python export_nnue_layerstacks_v2.py "
        "nnue_layerstacks_v2_best.pt ../checkpoints/nnue_layerstacks_v2_weights.bin")
endif()

add_custom_command(TARGET migoyugo_web POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${MGY_WEIGHTS}
    ${CMAKE_BINARY_DIR}/web/migoyugo_nnue_v2.bin)
