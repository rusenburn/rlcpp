#include <emscripten.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <sstream>
#include <iostream>

// Include necessary headers from the project
#include <games/migoyugo.hpp>
#include <players/g_player.hpp>
#include <common/state.hpp>
#include <common/exceptions.hpp>

// WASM-exported function to select a move
// Parameters:
// - actions_history: pointer to array of integers representing action history
// - history_length: number of actions in history
// - num_simulations: minimum number of MCTS simulations
// - duration_ms: time limit in milliseconds
// - min_ref_count: minimum reference count for G-RAVE
// - bias: exploration bias parameter
// - save_illegal_actions: whether to save illegal AMAF actions
// Returns: chosen action (integer), or -1 on error
extern "C" {

EMSCRIPTEN_KEEPALIVE
int select_move(const int* actions_history, int history_length,
                int num_simulations, int duration_ms,
                int min_ref_count, float bias, bool save_illegal_actions) {
    try {
        // Validate inputs
        if (history_length < 0 || history_length > 1000) {
            return -6; // Invalid history length
        }

        // Reconstruct game state from action history
        auto state = rl::games::MigoyugoState::initialize();

        // Apply each action in the history
        for (int i = 0; i < history_length; ++i) {
            int action = actions_history[i];
            if (action < 0 || action > 63) {
                return -7; // Invalid action
            }
            if (state->is_terminal()) {
                // If we reach terminal state but have more history, it's an error
                return -1;
            }
            state = state->step(action);
        }

        // Check if current state is terminal
        if (state->is_terminal()) {
            return -2; // Game already ended
        }

        // Create GPlayer with specified parameters
        rl::players::GPlayer player(
            num_simulations,
            std::chrono::milliseconds(duration_ms),
            min_ref_count,
            bias,
            save_illegal_actions
        );

        // Select and return the action
        return player.choose_action(state);

    } catch (const rl::common::IllegalActionException& e) {
        return -3; // Illegal action in history
    } catch (const rl::common::SteppingTerminalStateException& e) {
        return -4; // Attempted to step terminal state
    } catch (const std::exception& e) {
        return -5; // Generic error
    }
}

} // extern "C"
