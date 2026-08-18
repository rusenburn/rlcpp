cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(This onnxeval)
project(${This})

# ONNX Runtime is a binary release, not source, so it is neither a submodule like
# raylib/abseil nor a find_package on CMAKE_PREFIX_PATH like libtorch. Unpack the
# upstream tarball and point ORT_ROOT at it:
#
#   mkdir -p external/onnxruntime && curl -L \
#     https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz \
#     | tar xz -C external/onnxruntime --strip-components=1
#
# The tarball does ship lib/cmake/onnxruntime/onnxruntimeConfig.cmake, but that
# package is written for an *installed* layout - it points IMPORTED_LOCATION at
# lib64/ and its include dir at include/onnxruntime/, neither of which exists in
# the tarball. find_package would therefore succeed and then fail at link time,
# so the pieces are located by hand instead.
# Under Emscripten there is no native ONNX Runtime to find: inference goes
# through onnxruntime-web, which is JavaScript, so the backend is a JS bridge
# rather than a linked library. Everything above about ORT_ROOT applies to
# native builds only.
if(EMSCRIPTEN)
    # Built twice, once per async bridge.
    #
    # EM_ASYNC_JS in onnx_session_web.cpp compiles to a suspending import, and the
    # bridge has to be known at COMPILE time, not only at link - so the flag goes
    # on the library, PUBLIC so the executable inherits it. That is exactly why
    # one library cannot serve both: PUBLIC -sASYNCIFY would follow into the JSPI
    # executable and it would end up asking for both mechanisms at once.
    #
    # The C++ is identical in both. Emscripten's own test/core/test_em_async_js.c
    # is run against -sASYNCIFY and -sJSPI from the same source, so nothing in
    # onnx_session_web.cpp needs to know which one it is being built for.
    #
    # common/games/players are NOT duplicated: Asyncify instrumentation is a
    # link-time Binaryen pass, and none of those translation units use
    # EM_ASYNC_JS, so a single build of each serves both executables.
    function(add_onnxeval_web name async_flag)
        add_library(${name} STATIC
            src/onnx_session.cpp
            src/onnx_evaluator.cpp
            src/onnx_session_web.cpp)
        set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)

        target_include_directories(${name}
            PUBLIC
                $<INSTALL_INTERFACE:include>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )
        target_link_libraries(${name} PUBLIC common players)
        target_compile_options(${name} PUBLIC ${async_flag})
    endfunction()

    add_onnxeval_web(onnxeval      -sASYNCIFY)
    # -sJSPI, never -sASYNCIFY=2: emscripten's cmdline.py maps -sJSPI to
    # ASYNCIFY=2 *and* sets JSPI=1, and several of its JS paths are gated on
    # `#if JSPI` rather than on the ASYNCIFY value.
    add_onnxeval_web(onnxeval_jspi -sJSPI)
    return()
endif()

set(ORT_ROOT "${CMAKE_SOURCE_DIR}/external/onnxruntime"
    CACHE PATH "Root of an unpacked ONNX Runtime release")

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ORT_ROOT}/include" "${ORT_ROOT}/include/onnxruntime")
find_library(ORT_LIBRARY NAMES onnxruntime
    HINTS "${ORT_ROOT}/lib" "${ORT_ROOT}/lib64")

# Absent ONNX Runtime this target simply does not exist, and everything that uses
# it is guarded by if(TARGET onnxeval). A checkout without the tarball still
# configures and builds the rest of the project.
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(STATUS "onnxeval: ONNX Runtime not found under '${ORT_ROOT}' - skipping. "
                   "Set -DORT_ROOT=<unpacked release> to enable it.")
    return()
endif()
message(STATUS "onnxeval: using ONNX Runtime at ${ORT_LIBRARY}")

set(SourceFiles
    src/onnx_session.cpp
    src/onnx_evaluator.cpp
    src/onnx_session_ort.cpp
    )

set(HeaderFiles
    include/onnxeval/onnx_session.hpp
    include/onnxeval/onnx_evaluator.hpp
    include/onnxeval/onnx_session_ort.hpp
    )

add_library(${This} STATIC ${SourceFiles} ${HeaderFiles})
set_property(TARGET ${This} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${ORT_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        common
        players
    PRIVATE
        ${ORT_LIBRARY}
)

# libonnxruntime.so lives in the unpacked tarball, not on the loader path, so
# every executable that links this needs an rpath to it. Propagating it as an
# INTERFACE option means consumers get it from linking onnxeval and do not have
# to know where ORT_ROOT is.
if(NOT MSVC)
    get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
    target_link_options(${PROJECT_NAME} INTERFACE "-Wl,-rpath,${ORT_LIBRARY_DIR}")
endif()
