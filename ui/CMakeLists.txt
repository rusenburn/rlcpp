set(This ui)

project(${This})

set(SourceFiles 
    main.cpp
    players_utils.cpp
    damma/damma_ui_windows.cpp
    damma/damma_ui.cpp
    othello/othello_ui.cpp
    othello/othello_ui_windows.cpp
    santorini/santorini_tournament_ui.cpp
    santorini/santorini_ui.cpp
    santorini/santorini_ui_windows.cpp
    walls/walls_ui.cpp
    walls/walls_ui_windows.cpp
    IGameui.cpp
)

set(CUDNN_LIBRARY_PATH "C:/Program Files/NVIDIA/CUDNN/v8.5.0.96/lib")
set(CUDNN_INCLUDE_PATH "C:/Program Files/NVIDIA/CUDNN/v8.5.0.96/include")
find_package(Torch REQUIRED)


add_executable(${This} ${SourceFiles})
set_property(TARGET ${This} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${TORCH_LIBRARIES}
    raylib
    common
    players
    games
    deeplearning)
target_include_directories(${PROJECT_NAME} PRIVATE
 raylib
 deeplearning)


# by chatgpt
set_target_properties(${PROJECT_NAME} PROPERTIES
RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

 # The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${TORCH_DLLS}
                   $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)


# by chatgpt
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env "PATH=$<TARGET_FILE_DIR:${PROJECT_NAME}>;$ENV{PATH}" $<TARGET_FILE:${PROJECT_NAME}>
    )
endif()

add_custom_command(TARGET ${PROJECT_NAME}
                    PRE_BUILD
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                    ${CMAKE_BINARY_DIR}/checkpoints)

file (GLOB CHECKPOINTS "../checkpoints/*.pt")
# if(CHECKPOINTS)
# add_custom_command(TARGET ${PROJECT_NAME}
#                    PRE_BUILD
#                    COMMAND ${CMAKE_COMMAND} -E make_directory
#                    ${CMAKE_BINARY_DIR}/checkpoints)
# add_custom_command(TARGET ${PROJECT_NAME}
#                    PRE_BUILD
#                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                    ${CHECKPOINTS}
#                    ${CMAKE_BINARY_DIR}/checkpoints)
# endif(CHECKPOINTS)