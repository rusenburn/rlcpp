#ifndef RL_RUN_TRAIN_AI_CONSOLE_HPP_
#define RL_RUN_TRAIN_AI_CONSOLE_HPP_

#include <string>
#include <deeplearning/alphazero/networks/az.hpp>
#include <common/state.hpp>
#include "console.hpp"
namespace rl::run
{
    using IState = rl::common::IState;
    using IStatePtr = std::unique_ptr<IState>;
    using IAlphazeroNetwork = rl::deeplearning::alphazero::IAlphazeroNetwork;
    using IAlphazeroNetworkPtr = std::unique_ptr<IAlphazeroNetwork>;

    class TrainAIConsole : public IConsole
    {
    private:
        static constexpr int TICTACTOE = 0;
        static constexpr int OTHELLO = 1;
        static constexpr int ENGLISH_DRAUGHTS = 2;
        static constexpr int WALLS = 3;

        int state_choice_ {1};
        int n_iterations_{20};
        int n_episodes_{128};
        int n_sims_{200};
        int n_epochs_{4};
        int n_batches_{8};
        float lr_{2.5e-4};
        float critic_coef_{0.5};
        int n_testing_episodes_{32};
        std::string load_path_{""};
        std::string save_name_{"temp.pt"};

        // network settings
        int filters{128};
        int fc_dimensions{512};
        int blocks{5};

        void print_current_settings();
        void edit_settings();
        void train_ai();
        void edit_game_settings();

        IStatePtr get_state_ptr();
        IAlphazeroNetworkPtr get_network_ptr();

    public:
        TrainAIConsole(/* args */);
        ~TrainAIConsole();
        void run() override;
    };

} // namespace rl::run

#endif