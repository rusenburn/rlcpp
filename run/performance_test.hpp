#ifndef RL_RUN_PERFORMANCE_TEST_HPP_
#define RL_RUN_PERFORMANCE_TEST_HPP_

#include <chrono>
#include <memory>
#include <iostream>
#include <vector>
#include <common/state.hpp>
#include "console.hpp"


namespace rl::run
{

using IState = rl::common::IState;
using IStatePtr = std::unique_ptr<IState>;


class PerformanceTest :public IConsole
{
private:
    static constexpr int TIC_TAC_TOE_GAME = 0;
    static constexpr int OTHELLO_GAME = 1;
    static constexpr int ENGLISH_DRAUGHTS_GAME = 2;
    static constexpr int WALLS_GAME = 3;
    static constexpr int DAMMA_GAME = 4;
    static constexpr int SANTORINI_GAME = 5;
    static constexpr int GOBBLET_GAME = 6;
    static constexpr int MIGOYUGO_GAME = 7;
    int state_index_{ OTHELLO_GAME };
    IStatePtr get_state_ptr();
    int choose_action(const std::vector<bool>& masks) const;
    
    void edit_game_settings();
    void print_current_settings();
    void start();
public:
    void run() override;
    ~PerformanceTest()override;
};

} // namespace rl::run


#endif