#include <players/random_rollout_evaluator.hpp>

#include <common/exceptions.hpp>
#include <common/random.hpp>

namespace rl::players
{
    RandomRolloutEvaluator::RandomRolloutEvaluator(int n_game_actions)
        : n_game_actions_{n_game_actions}
    {
    }

    RandomRolloutEvaluator::~RandomRolloutEvaluator() = default;

    std::tuple<std::vector<float>, std::vector<float>> RandomRolloutEvaluator::evaluate(const std::vector<const rl::common::IState *> &state_ptrs)
    {
        std::vector<float> probs{};
        std::vector<float> values{};
        int n_states = static_cast<int>(state_ptrs.size());
        probs.reserve(n_game_actions_ * n_states);
        values.reserve(n_states);

        for (auto state_ptr : state_ptrs)
        {
            evaluate(state_ptr, probs, values);
        }
        return std::make_tuple(probs, values);
    }
    std::tuple<std::vector<float>, std::vector<float>> RandomRolloutEvaluator::evaluate(const rl::common::IState *state_ptrs)
    {
        std::vector<float> probs{};
        std::vector<float> values{};
        probs.reserve(n_game_actions_);
        values.reserve(1);
        evaluate(state_ptrs, probs, values);
        return std::make_tuple(probs, values);
    }
    std::tuple<std::vector<float>, std::vector<float>> RandomRolloutEvaluator::evaluate(const std::unique_ptr<rl::common::IState> &state_ptrs)
    {
        std::vector<float> probs{};
        std::vector<float> values{};
        probs.reserve(n_game_actions_);
        values.reserve(1);
        evaluate(state_ptrs.get(), probs, values);
        return std::make_tuple(probs, values);
    }

    void RandomRolloutEvaluator::evaluate(const rl::common::IState *state_ptr, std::vector<float> &probs, std::vector<float> &values) const
    {
        if (state_ptr->is_terminal())
        {
            rl::common::SteppingTerminalStateException("Evaluator trying to step a terminal state");
        }

        int starting_player = state_ptr->player_turn();
        std::vector<bool> masks = state_ptr->actions_mask();
        int n_legal_actions = 0;
        for (int action{0}; action < n_game_actions_; action++)
        {
            n_legal_actions += masks.at(action);
        }
        float legal_action_prob = 1.0f / float(n_legal_actions);
        for (int action{0}; action < n_game_actions_; action++)
        {
            probs.emplace_back(masks.at(action) ? legal_action_prob : 0.0f);
        }

        auto unique_state_ptr = std::unique_ptr<rl::common::IState>(nullptr);
        do
        {
            masks = state_ptr->actions_mask();
            int action = choose_action(masks);
            unique_state_ptr = state_ptr->step(action);
            state_ptr = unique_state_ptr.get();
        } while (!state_ptr->is_terminal());

        auto result = unique_state_ptr->get_reward();
        int last_player = unique_state_ptr->player_turn();
        if (last_player != starting_player)
        {
            result = -result;
        }
        values.emplace_back(result);
    }

    int RandomRolloutEvaluator::choose_action(const std::vector<bool> &masks) const
    {
        std::vector<int> legal_actions;
        int n_legal_actions = 0;
        for (auto action{0}; action < n_game_actions_; action++)
        {
            n_legal_actions += masks.at(action);
            if (masks.at(action))
            {
                legal_actions.push_back(action);
            }
        }

        int action_idx = rl::common::get(n_legal_actions);
        // int action_idx = rand() % n_legal_actions;
        int action = legal_actions[action_idx];
        return action;
    }

    // RandomRolloutEvaluator::RandomRolloutEvaluator(RandomRolloutEvaluator const &other)
    //     : n_game_actions_{other.n_game_actions_}
    // {
    // }
    std::unique_ptr<IEvaluator> RandomRolloutEvaluator::clone() const
    {
        return std::unique_ptr<RandomRolloutEvaluator>(new RandomRolloutEvaluator(*this));
    }

    std::unique_ptr<IEvaluator> RandomRolloutEvaluator::copy() const
    {
        return std::unique_ptr<RandomRolloutEvaluator>(new RandomRolloutEvaluator(*this));
    }
} // namespace rl::evaluators
