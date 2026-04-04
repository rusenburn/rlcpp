#include <iostream>
#include "main_console.hpp"
#include "concurrent_match_console.hpp"
#include "analyzer_console.hpp"
#include "performance_test.hpp"
namespace rl::run
{
MainConsole::MainConsole() = default;
MainConsole::~MainConsole() = default;
void MainConsole::run()
{
    std::cout << "Chose Option?\n";
    std::cout << "[0] Train AI \n";
    std::cout << "[1] Match \n";
    std::cout << "[2] Concurrent Match \n";
    std::cout << "[3] Analyzer \n";
    std::cout << "[4] Performance Test \n";
    int choice;
    std::cin >> choice;
    ConcurrentMatchConsole concurrent_match_console = ConcurrentMatchConsole();
    AnalyzerConsole ac = AnalyzerConsole();
    PerformanceTest ptc{};
    switch (choice)
    {
    case 0:
        train_ai_console_.run();
        break;
    case 1:
        match_console_.run();
        break;
    case 2:
        concurrent_match_console.run();
        break;
    case 3:
        ac.run();
        break;
    case 4:
        ptc.run();
        break;
        
    default:
        break;
    }
}
} // namespace rl::run
