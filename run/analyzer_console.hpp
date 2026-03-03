#ifndef RL_RUN_ANALYZER_CONSOLE_HPP_
#define RL_RUN_ANALYZER_CONSOLE_HPP_

#include <chrono>
#include <memory>
#include <iostream>
#include <vector>
#include <players/bandits/amcts2/concurrent_amcts.hpp>
#include <deeplearning/alphazero/networks/az.hpp>
#include <deeplearning/network_evaluator.hpp>
#include "console.hpp"
#include <common/match.hpp>


namespace rl::run
{
using IState = rl::common::IState;
using IStatePtr = std::unique_ptr<IState>;
using INetwork = rl::deeplearning::alphazero::IAlphazeroNetwork;
using INetworkPtr = std::unique_ptr<INetwork>;
using NetworkEvaluator = rl::deeplearning::NetworkEvaluator;


class AnalyzerConsole :public IConsole
{
public:
    ~AnalyzerConsole()override;
    void run() override;
private:
    static constexpr int TIC_TAC_TOE_GAME = 0;
    static constexpr int OTHELLO_GAME = 1;
    static constexpr int ENGLISH_DRAUGHTS_GAME = 2;
    static constexpr int WALLS_GAME = 3;
    static constexpr int DAMMA_GAME = 4;
    static constexpr int SANTORINI_GAME = 5;
    static constexpr int GOBBLET_GAME = 6;
    static constexpr int MIGOYUGO_GAME = 7;
    int state_index_{ MIGOYUGO_GAME };
    void get_match();
    void get_actions(const std::vector<int>& actions);
    IStatePtr get_state_ptr();
    INetworkPtr get_network_ptr(int filters, int fc_dims, int blocks, const std::string& load_name);
    std::unique_ptr<rl::players::IEvaluator> get_network_evaluator_ptr(INetworkPtr& network_ptr);
    std::chrono::duration<int, std::milli> duration{ 25000 };
    int n_filters{ 128 };
    int fc_dims{ 512 };
    int blocks{ 5 };
    std::string load_name{ "migoyugo_strongest_480.pt" };
};
} // namespace rl::run


#endif