#include <deeplearning/alphazero/alphazero.hpp>
#include <deeplearning/alphazero/networks/smallnn.hpp>
#include <games/othello.hpp>
#include "main_console.hpp"

int main(int argc, char const* argv[])
{

    try
    {
        rl::run::MainConsole console{};
        console.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    catch (const char* arr)
    {
        std::cerr << arr << "\n";
    }

    // run_match(2,5000,10);
    return 0;
}
