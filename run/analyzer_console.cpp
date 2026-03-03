#include "analyzer_console.hpp"
#include <iostream>
#include <vector>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <players/bandits/amcts2/concurrent_amcts.hpp>
#include <games/tictactoe.hpp>
#include <games/othello.hpp>
#include <games/english_draughts.hpp>
#include <games/walls.hpp>
#include <games/damma.hpp>
#include <games/santorini.hpp>
#include <games/gobblet_goblers.hpp>
#include <games/migoyugo.hpp>
#include <string>
#include <iostream>
#include <vector>
namespace rl::run
{

std::string get_coord_from_id(int action_id) {
    if (action_id < 0 || action_id > 63) return "??";

    // 1. Calculate Row and Column (Inverse of row * 8 + column)
    int row = action_id / 8;
    int col = action_id % 8;

    // 2. Convert Row back to Rank (8 - row)
    int rank = 8 - row;

    // 3. Convert Column back to Letter (0 -> 'A', 1 -> 'B', etc.)
    char file = static_cast<char>('a' + col);

    // 4. Build the string (e.g., 'a' + '8')
    return std::string(1, file) + std::to_string(rank);
}
AnalyzerConsole::~AnalyzerConsole() = default;
void AnalyzerConsole::run() {
    int choice = 0;
    // do
    // {
    //     std::cout << "Do you want to run current settings?\n";
    //     std::cout << "[0] Yes, run\n";
    //     std::cout << "[1] No, I want to edit settings\n";
    //     std::cout << "[2] No, quit match\n";
    //     std::cin >> choice;
    //     if (choice == 1)
    //     {
    //         edit_settings();
    //     }
    //     else if (choice == 2)
    //     {
    //         return;
    //     }
    // } while (choice != 0);


    get_match();
}

void AnalyzerConsole::get_match() {
    std::vector<int> action_sequence;
    int choice;

    std::cout << "Enter actions (space or enter separated). Enter a negative number to finish:" << std::endl;

    // This will keep reading until a non-integer is entered or the loop is broken
    while (std::cin >> choice) {
        if (choice < 0) {
            // If negative, call get_actions with the accumulated vector and exit
            get_actions(action_sequence);
            break;
        }

        // Add valid action to the vector
        action_sequence.push_back(choice);
    }
}


std::string get_coord(int id) {
    char file = 'a' + (id % 8);
    int rank = 8 - (id / 8);
    return std::string(1, file) + std::to_string(rank);
}

void AnalyzerConsole::get_actions(const std::vector<int>& actions) {

    auto state_ptr = get_state_ptr(); // Initial state
    auto network_ptr = get_network_ptr(n_filters, fc_dims, blocks, load_name);
    auto evaluator_ptr = get_network_evaluator_ptr(network_ptr);

    // Initialize AMCTS
    rl::players::ConcurrentAmcts amcts(
        state_ptr->get_n_actions(),
        std::move(evaluator_ptr->copy()),
        2.0f, 1.0f, 8, 0.0f, -1.0f, 1.0f, -1.0f
    );

    // This vector will hold the actual objects to keep them in memory
    std::vector<std::unique_ptr<rl::common::IState>> history;
    // This vector will hold the raw pointers for search_multiple
    std::vector<const rl::common::IState*> state_ptrs_to_search{};

    std::cout << "\n--- Stepping through " << actions.size() << " Actions ---" << std::endl;

    for (int action : actions) {
        if (!state_ptr->is_terminal()) {
            // 1. Move the current state into our history to keep it alive
            history.push_back(state_ptr->clone());

            // 2. Add the raw pointer from the cloned state to our search list
            state_ptrs_to_search.push_back(history.back().get());

            // 3. Step to the next state
            state_ptr = state_ptr->step(action);
            std::cout << action << " ";
        }
        else {
            std::cout << "(" << action << " skipped: Terminal) ";
        }
    }

    // Call search_multiple with the full list of states encountered
    auto d = duration;
    auto [all_probs, all_v] = amcts.search_multiple(state_ptrs_to_search, 1, d);

    // // Loop through the values (all_v) and print them 1 by 1
    std::cout << "\n--- Search Results (Win Values) ---" << std::endl;
    // for (size_t i = 0; i < all_v.size(); ++i) {
    //     // Output formatting: Index starts at 0 (Initial State)
    //     std::cout << "State " << i << " Value: " << static_cast<int>( all_v[i] *100) << std::endl;
    // }

    // std::cout << "-----------------------------------" << std::endl;

    for (size_t i = 0; i < all_v.size(); ++i) {
        int step = (i / 2) + 1;
        // 1. Get the Value for this state
        float state_val = all_v[i];
        int action = actions.at(i);

        // 2. Filter and collect only actions with prob > 0
        const std::vector<float>& probs = all_probs[i];
        std::vector<std::pair<int, float>> valid_moves;

        for (int j = 0; j < (int)probs.size(); ++j) {
            if (probs[j] > 1e-6f) { // Use a small epsilon instead of 0
                valid_moves.push_back({ j, probs[j] });
            }
        }

        // 3. Sort by probability (Descending)
        std::sort(valid_moves.begin(), valid_moves.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
            });

        // 4. Print results (State Value + Top Moves)
        std::cout << "Turn " << step << (i % 2 == 0 ? " White" : " Black") << " Val: " << (state_val * 100 * (i % 2 == 0 ? 1 : -1)) << " Played: " << get_coord(action) << " Prob: " << (probs.at(action) * 100) << "%" << std::endl;

        // Take Top 5 or whatever is available
        size_t limit = std::min<size_t>(5, valid_moves.size());
        for (size_t k = 0; k < limit; ++k) {
            int action_id = valid_moves[k].first;
            float probability = valid_moves[k].second;

            // Use the function here to show "e4" instead of "28"
            std::cout << "  Move: " << get_coord(action_id)
                << " (ID: " << action_id << ")"
                << " Prob: " << (probability * 100) << "%" << std::endl;
        }
    }
}


IStatePtr AnalyzerConsole::get_state_ptr()
{
    switch (state_index_)
    {
    case TIC_TAC_TOE_GAME:
        return rl::games::TicTacToeState::initialize();
        break;
    case OTHELLO_GAME:
        return rl::games::OthelloState::initialize();
        break;
    case ENGLISH_DRAUGHTS_GAME:
        return rl::games::EnglishDraughtState::initialize();
        break;
    case WALLS_GAME:
        return rl::games::WallsState::initialize();
        break;
    case DAMMA_GAME:
        return rl::games::DammaState::initialize();
        break;
    case SANTORINI_GAME:
        return rl::games::SantoriniState::initialize();
        break;
    case GOBBLET_GAME:
        return rl::games::GobbletGoblersState::initialize();
        break;
    case MIGOYUGO_GAME:
        return rl::games::MigoyugoState::initialize();
        break;
    default:
        throw "";
        break;
    }
}

INetworkPtr AnalyzerConsole::get_network_ptr(int filters, int fc_dims, int blocks, const std::string& load_name)
{
    auto state_pr = get_state_ptr();
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state_pr->get_observation_shape(), state_pr->get_n_actions(),
        filters, fc_dims, blocks);
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    const std::string folder_name = "../checkpoints";
    std::filesystem::path folder(folder_name);
    std::filesystem::path file_path;
    file_path = folder / load_name;
    network_ptr->load(file_path.string());
    network_ptr->to(device);
    int n_examples = 1;
    int n_channels = state_pr->get_observation_shape()[0];
    int n_rows = state_pr->get_observation_shape()[1];
    int n_cols = state_pr->get_observation_shape()[2];

    network_ptr->forward(torch::randn(n_examples * n_channels * n_rows * n_cols).to(device).reshape({ n_examples, n_channels, n_rows, n_cols }));
    return network_ptr;
}

std::unique_ptr<rl::players::IEvaluator>  AnalyzerConsole::get_network_evaluator_ptr(INetworkPtr& network_ptr)
{
    auto state_ptr = get_state_ptr();
    return std::make_unique<rl::deeplearning::NetworkEvaluator>(network_ptr->copy(), state_ptr->get_n_actions(), state_ptr->get_observation_shape());
}

} // namespace rl::run


