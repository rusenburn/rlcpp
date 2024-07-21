#include <raylib.h>
#include "othello/othello_ui.hpp"
#include "walls/walls_ui.hpp"
#include "damma/damma_ui.hpp"
#include "santorini/santorini_ui.hpp"
#include "santorini/santorini_tournament_ui.hpp"
#include <string>

int main(int argc, char const *argv[])
{
    std::string a;
    std::getline(std::cin, a);
    std::cout << a;
    if (a != "o")
    {
        return 0;
    }
    Color GREY = {29, 29, 29, 255};

    constexpr int WINDOW_W = 720;
    constexpr int WINDOW_H = 720;
    constexpr int FPS = 12;
    
    rl::ui::SantoriniTournamentUI ui{WINDOW_W, WINDOW_H};

    InitWindow(WINDOW_W, WINDOW_H, "Reinforcement Learning");
    ui.init();
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
