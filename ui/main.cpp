#include <raylib.h>
#include "othello/othello_ui.hpp"
#include "walls/walls_ui.hpp"


int main(int argc, char const *argv[])
{
    Color GREY = {29, 29, 29, 255};

    constexpr int WINDOW_W = 720;
    constexpr int WINDOW_H = 720;
    constexpr int FPS = 12;
    
    rl::ui::WallsUi ui{WINDOW_W, WINDOW_H};

    InitWindow(WINDOW_W, WINDOW_H, "Reinforcement Learning");
    SetTargetFPS(FPS);

    while (WindowShouldClose() == false)
    {

        ui.handle_events();
        BeginDrawing();
        ClearBackground(GREY);
        ui.draw_game();
        EndDrawing();
    }
    CloseWindow();
    return 0;
}
