#include <nnue/nnue_mcts_player.hpp>
#include <games/migoyugo_light.hpp>
#include <nnue/nnue_mcts.hpp>
#include <common/random.hpp>
namespace rl::players
{
NNUEMctsPlayer::NNUEMctsPlayer(NNUEModel model, int n_game_actions, int minimum_simulations, std::chrono::duration<int, std::milli> duration_in_millis, float cpuct)

    :model_(model),
    n_game_actions_{ n_game_actions },
    minimum_simulations_{ minimum_simulations },
    duration_in_millis_{duration_in_millis},
    cpuct_{ cpuct }
{
}




rl::players::NNUEMctsPlayer::~NNUEMctsPlayer() = default;

int NNUEMctsPlayer::choose_action(const std::unique_ptr<rl::common::IState>& state_ptr)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    auto light_state = rl::games::MigoyugoLightState::from_short(state_ptr->to_short());

    std::array<int16_t, 256> acc_w, acc_b;
    rl::games::NNUEUpdate initial_features;
    light_state->get_active_features(initial_features);

    auto features = light_state->calculate_feature_weight();
    float temperature = features <= 20 ? 1.0f : 0.0f;
    auto mcts = rl::nnue::MCTS(light_state->get_n_actions(), cpuct_, temperature);

    auto probs = mcts.search(light_state.get(), model_, minimum_simulations_, duration_in_millis_);

    float p = rl::common::get();

    float remaining_prob = p;

    int action = 0;

    int last_action = n_game_actions_ - 1;

    // keep decreasing remaining probs until it is below zero or only 1 action remains
    while ((action < last_action) && ((remaining_prob -= probs.at(action)) >= 0))
    {
        action++;
    }
    return action;
}


}





