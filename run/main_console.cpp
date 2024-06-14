#include <iostream>
#include "main_console.hpp"

namespace rl::run
{
    MainConsole::MainConsole() = default;
    MainConsole::~MainConsole() = default;
    void MainConsole::run()
    {
        std::cout << "Chose Option?\n";
        std::cout << "[0] Train AI \n";
        std::cout << "[1] Match \n";
        int choice;
        std::cin >> choice;

        switch (choice)
        {
        case 0:
            train_ai_console_.run();
            break;
        case 1:
            match_console_.run();
            break;
        default:
            break;
        }
    }
} // namespace rl::run
