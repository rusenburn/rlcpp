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

# ---------------------------------------------------------------------------
# migoyugo_az - the AlphaZero bot: Amcts2 over MigoyugoState, with the residual
# network evaluated by onnxruntime-web on WebGPU.
#
# A separate module from migoyugo_wasm on purpose: it needs an async bridge so
# the synchronous Amcts2::search() can await onnxruntime-web's Promise, and that
# bridge instruments or wraps everything linked into it. Keeping the two apart
# means the NNUE alpha-beta engine keeps the codegen it has today.
#
# Built TWICE, once per bridge, and az_worker.js loads whichever the browser
# supports:
#
#   migoyugo_az       -sASYNCIFY  rewrites the wasm to unwind and rewind its own
#                                 stack. Works everywhere, costs size and a stack
#                                 copy on each of the ~150 suspends per move.
#   migoyugo_az_jspi  -sJSPI      the VM switches stacks natively - no rewriting.
#                                 Chrome 137+, Firefox 139+; Safari not yet.
#
# Everything except the bridge is identical, so the options are shared below and
# only the differing flags are per-target.
# ---------------------------------------------------------------------------

set(MGY_AZ_EXPORTS "_malloc,_free,_mgy_az_init,_mgy_az_new_game,_mgy_az_play,_mgy_az_undo,_mgy_az_load_moves,_mgy_az_bot_move,_mgy_az_bot_suggest,_mgy_az_set_time_ms,_mgy_az_set_batch,_mgy_az_snapshot,_mgy_az_snapshot_size,_mgy_az_probs,_mgy_az_evaluation,_mgy_az_last_evaluations")

set(MGY_AZ_COMMON_LINK_OPTIONS
    -sMODULARIZE=1
    # The same factory name in both builds. Only one glue file is ever loaded, so
    # there is no collision, and az_worker.js keeps a single init path.
    -sEXPORT_NAME=createMigoyugoAz
    -sENVIRONMENT=worker

    -sSTACK_SIZE=4MB
    # Only the MCTS tree lives in linear memory; the 40 MB of weights stay on
    # the JS side inside onnxruntime-web.
    -sINITIAL_MEMORY=64MB
    -sALLOW_MEMORY_GROWTH=1
    -sMALLOC=emmalloc

    -sFILESYSTEM=0
    -sEXIT_RUNTIME=0
    -sINVOKE_RUN=0

    "-sEXPORTED_FUNCTIONS=[${MGY_AZ_EXPORTS}]"
    # HEAPF32 as well as HEAPU8: the observation and policy buffers the JS
    # bridge reads and writes are float32.
    #
    # ccall is not optional, under EITHER bridge. Under Asyncify a plain
    # _mgy_az_bot_move() returns a bogus value the instant the search first
    # suspends, long before a move exists. Under JSPI the export does return a
    # real Promise, but ccall(..., {async: true}) works there too
    # (emscripten src/lib/libccall.js), so one call form serves both.
    -sEXPORTED_RUNTIME_METHODS=[HEAPU8,HEAPF32,ccall]
)

# migoyugo_az - Asyncify. The portable one; also the fallback when JSPI is
# missing or fails.
set(AzThis migoyugo_az_wasm)

add_executable(${AzThis} migoyugo_az_wasm.cpp)
target_link_libraries(${AzThis} PRIVATE onnxeval players games common)
set_target_properties(${AzThis} PROPERTIES OUTPUT_NAME migoyugo_az)

target_link_options(${AzThis} PRIVATE
    ${MGY_AZ_COMMON_LINK_OPTIONS}

    # EM_ASYNC_JS in onnx_session_web.cpp suspends the wasm stack until
    # onnxruntime-web resolves; without this the module fails to link with
    # "ASYNCIFY is not enabled".
    -sASYNCIFY
    # The suspended stack is shallow - Amcts2::search -> OnnxEvaluator::evaluate
    # -> run -> JS, with the tree recursion already unwound - but the default
    # 4 KB unwind buffer is tight enough that overflowing it aborts at runtime
    # rather than at link time. 64 KB costs nothing.
    -sASYNCIFY_STACK_SIZE=65536
)

set_target_properties(${AzThis} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/web)

# migoyugo_az_jspi - same source, native stack switching.
set(AzJspiThis migoyugo_az_jspi_wasm)

add_executable(${AzJspiThis} migoyugo_az_wasm.cpp)
target_link_libraries(${AzJspiThis} PRIVATE onnxeval_jspi players games common)
set_target_properties(${AzJspiThis} PROPERTIES OUTPUT_NAME migoyugo_az_jspi)

target_link_options(${AzJspiThis} PRIVATE
    ${MGY_AZ_COMMON_LINK_OPTIONS}

    -sJSPI
    # Mandatory, not tuning. Only `main` is promised by default; any other export
    # that suspends must be listed here or it is never wrapped with
    # WebAssembly.promising and traps the moment it tries to suspend. Both are
    # named explicitly rather than with a wildcard so a rename fails the build.
    -sJSPI_EXPORTS=mgy_az_bot_move,mgy_az_bot_suggest
    # -sJSPI unconditionally warns that it is experimental (tools/link.py). It is
    # shipped in Chrome and Firefox and standardised; silence the per-build noise.
    -Wno-experimental

    # No ASYNCIFY_STACK_SIZE here, and no other ASYNCIFY_* knob: they apply only
    # to ASYNCIFY=1. There is no unwind buffer to size - the VM owns the stack.
)

set_target_properties(${AzJspiThis} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/web)

set(MGY_AZ_WEB_FILES
    web/az.html
    web/az_worker.js
    web/az_snapshot.js)

set(MGY_AZ_COPY_COMMANDS "")
foreach(f ${MGY_AZ_WEB_FILES})
    get_filename_component(name ${f} NAME)
    list(APPEND MGY_AZ_COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/${f}
        ${CMAKE_BINARY_DIR}/web/${name})
endforeach()

add_custom_target(migoyugo_az_web ALL
    ${MGY_AZ_COPY_COMMANDS}
    COMMENT "Copying AlphaZero web assets to ${CMAKE_BINARY_DIR}/web"
    VERBATIM)
add_dependencies(migoyugo_az_web ${AzThis} ${AzJspiThis})

# Unlike the NNUE weights this is a warning, not a fatal error: the .onnx is
# 40 MB and only the AlphaZero page needs it, so a missing one must not block
# somebody building just the NNUE engine. The page reports the 404 itself.
set(MGY_AZ_MODEL ${CMAKE_SOURCE_DIR}/checkpoints/migoyugo_az.onnx)
if(EXISTS ${MGY_AZ_MODEL})
    add_custom_command(TARGET migoyugo_az_web POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${MGY_AZ_MODEL}
        ${CMAKE_BINARY_DIR}/web/migoyugo_az.onnx)
else()
    message(WARNING
        "Missing ${MGY_AZ_MODEL} - the AlphaZero page will 404 on its network.\n"
        "Export it first:\n"
        "  cd scripts && python export_az_onnx.py "
        "../checkpoints/migoyugo_strongest_900.pt ../checkpoints/migoyugo_az.onnx")
endif()
