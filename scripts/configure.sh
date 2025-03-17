# !/bin/bash

cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_PREFIX_PATH:STRING=C:/libtorch-win-shared-with-deps-2.2.1+cu118/libtorch -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE --no-warn-unused-cli -SC:/Users/lucif/source/cpp/rlcpp -Bc:/Users/lucif/source/cpp/rlcpp/build/Release -G "Visual Studio 17 2022"