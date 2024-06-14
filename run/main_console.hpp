#ifndef RL_RUN_MAIN_CONSOLE_HPP_
#define RL_RUN_MAIN_CONSOLE_HPP_

#include "console.hpp"
#include "train_ai_console.hpp"
#include "match_console.hpp"
namespace rl::run
{
    class MainConsole : public IConsole
    {
    private:
        TrainAIConsole train_ai_console_{};
        MatchConsole match_console_{};
    public:
        MainConsole(/* args */);
        ~MainConsole();
        void run() override;
    };
} // namespace rl::run

#endif