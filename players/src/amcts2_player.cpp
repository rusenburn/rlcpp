#include <players/amcts2_player.hpp>
#include <players/bandits/amcts2/amcts2.hpp>
#include <common/random.hpp>
#include <cassert>

namespace rl::players
{
Amcts2Player::Amcts2Player(
    int n_game_actions,
    std::unique_ptr<IEvaluator> evaluator_ptr,
    int minimum_simulations,
    std::chrono::duration<int, std::milli> duration_in_millis,
    float temperature,
    float cpuct,
    int max_async_simulations,
    float dirichlet_epsilon,
    float dirichlet_alpha,
    float default_visits,
    float default_wins)

    : n_game_actions_{ n_game_actions },
    evaluator_ptr_{ std::move(evaluator_ptr) },
    minimum_simulations_{ minimum_simulations },
    duration_in_millis_{ duration_in_millis },
    temperature_{ temperature },
    cpuct_{ cpuct },
    max_async_simulations_{ max_async_simulations },
    dirichlet_epsilon_{dirichlet_epsilon},
    dirichlet_alpha_{dirichlet_alpha},
    default_visits_{ default_visits },
    default_wins_{ default_wins }
{
}


Amcts2Player::~Amcts2Player() = default;

int Amcts2Player::choose_action(const std::unique_ptr<rl::common::IState>& state_ptr)
{
    auto mcts = Amcts2(n_game_actions_, evaluator_ptr_->copy(), cpuct_, temperature_, max_async_simulations_,dirichlet_epsilon_,dirichlet_alpha_, default_visits_, default_wins_);
    std::vector<float> probs = mcts.search(state_ptr.get(), minimum_simulations_, duration_in_millis_);

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
} // namespace rl::players

#include <players/bandits/amcts2/concurrent_amcts.hpp>

namespace rl::players
{
ConcurrentPlayer::ConcurrentPlayer(
    int n_game_actions,
    std::unique_ptr<IEvaluator> evaluator_ptr,
    int minimum_simulations,
    std::chrono::duration<int, std::milli> duration_in_millis,
    float temperature,
    float cpuct,
    int max_async_simulations ,
    float dirichlet_epsilon,
    float dirichlet_alpha,
    float default_visits ,
    float default_wins 
)
    :n_game_actions_{ n_game_actions },
    evaluator_ptr_{ std::move(evaluator_ptr) },
    minimum_simulations_{ minimum_simulations },
    duration_in_millis_{ duration_in_millis },
    temperature_{ temperature },
    cpuct_{ cpuct },
    max_async_simulations_{ max_async_simulations },
    dirichlet_epsilon_{dirichlet_epsilon},
    dirichlet_alpha_{dirichlet_alpha_},
    default_visits_{ default_visits },
    default_wins_{ default_wins }
{

}

ConcurrentPlayer::~ConcurrentPlayer() = default;

std::vector<int> ConcurrentPlayer::choose_actions(const std::vector<const rl::common::IState*>& states_ptrs_ref)
{
    auto amcts = ConcurrentAmcts(n_game_actions_, evaluator_ptr_->copy(), cpuct_, temperature_, max_async_simulations_, dirichlet_epsilon_,dirichlet_alpha_,default_visits_, default_wins_);
    auto& [all_probs, all_values] = amcts.search_multiple(states_ptrs_ref, minimum_simulations_, duration_in_millis_);
    const int n_states = states_ptrs_ref.size();
    std::vector<int> actions{};
    for (int i = 0;i < n_states;i++)
    {
        auto& probs = all_probs.at(i);
        float p = rl::common::get();

        float remaining_prob = p;

        int action = 0;

        int last_action = n_game_actions_ - 1;

        // keep decreasing remaining probs until it is below zero or only 1 action remains
        while ((action < last_action) && ((remaining_prob -= probs.at(action)) >= 0))
        {
            action++;
        }

        actions.push_back(action);
    }
    return actions;
}

int ConcurrentPlayer::choose_action(const std::unique_ptr<rl::common::IState>& state_ptr)
{
    std::vector<const rl::common::IState*> state_ptrs = { {state_ptr.get()} };
    auto all_actions = choose_actions(state_ptrs);
    assert(all_actions.size() == 1);
    return all_actions[0];
}




} // namespace rl::players


