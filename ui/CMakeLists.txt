set(This ui)

project(${This})

set(SourceFiles 
    main.cpp
    othello/othello_ui.cpp
    othello/othello_ui_windows.cpp
    walls/walls_ui.cpp
    walls/walls_ui_windows.cpp
    IGameui.cpp
)

add_executable(${This} ${SourceFiles})
set_property(TARGET ${This} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME} PUBLIC
    raylib
    common
    players
    games)
target_include_directories(${PROJECT_NAME} PRIVATE raylib)
