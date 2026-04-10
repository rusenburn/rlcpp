#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <games/migoyugo.hpp>
#include <games/migoyugo_light.hpp>
#include "NNUEModel.hpp" // Your struct and evaluate_nnue function
#include <torch/script.h> // LibTorch
#include <deeplearning/network_evaluator.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <algorithm>
#include <cstdint>
#include <array>
#include <common/random.hpp>

int sample_action(const std::vector<float>& probs, int game_idx, int n_actions, float temp) {
    // Calculate start of this game's probabilities in the batch vector
    int start_idx = game_idx * n_actions;

    // Apply Temperature for diversity
    std::vector<float> adjusted_probs;
    adjusted_probs.reserve(n_actions);
    float sum = 0.0f;

    for (int i = 0; i < n_actions; ++i) {
        float p = std::pow(probs[start_idx + i], 1.0f / temp);
        adjusted_probs.push_back(p);
        sum += p;
    }

    // Weighted Random Selection
    float r = rl::common::get() * sum;
    for (int i = 0; i < n_actions; ++i) {
        if ((r -= adjusted_probs[i]) <= 0) return i;
    }
    return n_actions - 1;
}

int choose_action(const std::vector<float>& probs, int n_game_actions)
{
    float p = rl::common::get();

    float remaining_prob = p;

    int action = 0;

    int last_action = n_game_actions - 1;

    // keep decreasing remaining probs until it is below zero or only 1 action remains
    while ((action < last_action) && ((remaining_prob -= probs.at(action)) >= 0))
    {
        action++;
    }

    return action;
}
int choose_action(const std::vector<bool>& masks, int n_game_actions)
{
    std::vector<int> legal_actions;
    int n_legal_actions = 0;
    for (auto action{ 0 }; action < n_game_actions; action++)
    {
        n_legal_actions += masks[action];
        if (masks[action])
        {
            legal_actions.push_back(action);
        }
    }

    int action_idx = rl::common::get(n_legal_actions);
    int action = legal_actions[action_idx];
    return action;
}


// Clipped ReLU for NNUE: f(x) = clamp(x / 128, 0, 127)
// We return int32 to prevent overflow in the next layer's summation
inline int32_t activation(int16_t x) {
    return std::clamp(static_cast<int32_t>(x), 0, 127);
}

float evaluate_nnue(const int16_t* accumulator, const NNUEModel& model) {
    // --- Layer 2: 256 -> 16 ---
    int32_t l2_out[16];
    for (int i = 0; i < 16; ++i) {
        int32_t sum = model.l2_bias[i];
        for (int j = 0; j < 256; ++j) {
            sum += model.l2_weights[i][j] * activation(accumulator[j]);
        }
        l2_out[i] = std::clamp(sum >> 7, 0, 127); // Re-quantize for L3
    }

    // --- Layer 3: 16 -> 32 ---
    int32_t l3_out[32];
    for (int i = 0; i < 32; ++i) {
        int32_t sum = model.l3_bias[i];
        for (int j = 0; j < 16; ++j) {
            sum += model.l3_weights[i][j] * l2_out[j];
        }
        l3_out[i] = std::clamp(sum >> 7, 0, 127);
    }

    // --- Output Layer: 32 -> 1 ---
    int32_t final_sum = model.out_bias;
    for (int i = 0; i < 32; ++i) {
        final_sum += model.out_weights[i] * l3_out[i];
    }

    // Final Scaling back to float (-1.0 to 1.0 range)
    // Since we scaled by 128 three times in the weights and biases,
    // we divide by (128^3) to get the original scale.
    return static_cast<float>(final_sum) / (128.0f * 128.0f);
}

float evaluate_nnue2(const std::array<int16_t, 256>& accumulator, const NNUEModel& model) {
    // --- Layer 2: 256 -> 16 ---
    std::array<int32_t, 16> l2_out;

    for (size_t i = 0; i < 16; ++i) {
        int32_t sum = model.l2_bias[i];

        // Accumulator is Layer 1 output
        for (size_t j = 0; j < 256; ++j) {
            sum += static_cast<int32_t>(model.l2_weights[i][j]) * activation(accumulator[j]);
        }

        // Re-quantize for Layer 3 (Shift 7 bits and clamp)
        l2_out[i] = std::clamp(sum >> 7, 0, 127);
    }

    // --- Layer 3: 16 -> 32 ---
    std::array<int32_t, 32> l3_out;

    for (size_t i = 0; i < 32; ++i) {
        int32_t sum = model.l3_bias[i];

        for (size_t j = 0; j < 16; ++j) {
            sum += static_cast<int32_t>(model.l3_weights[i][j]) * l2_out[j];
        }

        l3_out[i] = std::clamp(sum >> 7, 0, 127);
    }

    // --- Output Layer: 32 -> 1 ---
    int32_t final_sum = model.out_bias;

    for (size_t i = 0; i < 32; ++i) {
        final_sum += static_cast<int32_t>(model.out_weights[i]) * l3_out[i];
    }

    // Final Scaling: Divide by 128^2 to reach parity with your Torch model
    return static_cast<float>(final_sum) / (128.0f * 128.0f);
}


float evaluate_nnue_simd(const std::array<int16_t, 256>& accumulator, const NNUEModel& model) {
    // 1. Pre-calculate Activated Layer 1 (Clipped ReLU)
    // We do this once so we don't have to clamp inside the nested loops.
    alignas(32) std::array<int16_t, 256> activated_l1;
    for (size_t i = 0; i < 256; ++i) {
        activated_l1[i] = static_cast<int16_t>(std::clamp<int32_t>(accumulator[i], 0, 127));
    }

    // --- Layer 2: 256 -> 16 ---
    alignas(32) std::array<int32_t, 16> l2_out;
    for (size_t i = 0; i < 16; ++i) {
        // Initialize sum with bias
        __m128i sum_v = _mm_setzero_si128();

        // Process 256 inputs in chunks of 8 (128-bit SSE)
        for (size_t j = 0; j < 256; j += 8) {
            // Load 8 weights and 8 activated inputs
            __m128i weights = _mm_load_si128((__m128i*) & model.l2_weights[i][j]);
            __m128i inputs = _mm_load_si128((__m128i*) & activated_l1[j]);

            // _mm_madd_epi16: (w0*i0 + w1*i1), (w2*i2 + w3*i3)... 
            // Result is 4 int32s in one register
            __m128i madd = _mm_madd_epi16(weights, inputs);
            sum_v = _mm_add_epi32(sum_v, madd);
        }

        // Horizontal sum of the 4 ints in sum_v
        alignas(16) int32_t temp_sums[4];
        _mm_store_si128((__m128i*)temp_sums, sum_v);
        int32_t total_sum = model.l2_bias[i] + temp_sums[0] + temp_sums[1] + temp_sums[2] + temp_sums[3];

        l2_out[i] = std::clamp(total_sum >> 7, 0, 127);
    }

    // --- Layer 3: 16 -> 32 ---
    // (16 inputs is small, but we can still SSE it)
    alignas(32) std::array<int32_t, 32> l3_out;
    for (size_t i = 0; i < 32; ++i) {
        __m128i sum_v = _mm_setzero_si128();

        // Convert l2_out (int32) to int16 temporarily for madd instruction
        // or just do standard math since 16 iterations is tiny.
        int32_t total_sum = model.l3_bias[i];
        for (size_t j = 0; j < 16; ++j) {
            total_sum += model.l3_weights[i][j] * l2_out[j];
        }
        l3_out[i] = std::clamp(total_sum >> 7, 0, 127);
    }

    // --- Output Layer: 32 -> 1 ---
    int32_t final_sum = model.out_bias;
    for (size_t i = 0; i < 32; i += 4) {
        // Simple 32-bit math for the final layer
        for (size_t k = 0; k < 4; ++k) {
            final_sum += model.out_weights[i + k] * l3_out[i + k];
        }
    }

    return static_cast<float>(final_sum) / (128.0f * 128.0f);
}


void update_nnue(std::array<int16_t, 256>& acc_white,
    std::array<int16_t, 256>& acc_black,
    const rl::games::NNUEUpdate& update,
    const NNUEModel& model)
{
    // Update White Accumulator
    for (int fid : update.white_added) {
        for (int j = 0; j < 256; ++j) acc_white[j] += model.l1_weights[fid][j];
    }
    for (int fid : update.white_removed) {
        for (int j = 0; j < 256; ++j) acc_white[j] -= model.l1_weights[fid][j];
    }

    // Update Black Accumulator
    for (int fid : update.black_added) {
        for (int j = 0; j < 256; ++j) acc_black[j] += model.l1_weights[fid][j];
    }
    for (int fid : update.black_removed) {
        for (int j = 0; j < 256; ++j) acc_black[j] -= model.l1_weights[fid][j];
    }
}
// float evaluate_nnue(const int16_t* accumulator, const NNUEModel& model) {
//     // --- L2: 256 -> 16 ---
//     int32_t l2_out[16];
//     for (int i = 0; i < 16; ++i) {
//         int32_t sum = model.l2_bias[i]; // Already scaled by 128*128
//         for (int j = 0; j < 256; ++j) {
//             sum += model.l2_weights[i][j] * activation(accumulator[j]);
//         }
//         l2_out[i] = std::clamp(sum >> 7, 0, 128); // Bring back to 128-scale
//     }

//     // --- L3: 16 -> 32 ---
//     int32_t l3_out[32];
//     for (int i = 0; i < 32; ++i) {
//         int32_t sum = model.l3_bias[i]; // Already scaled by 128*128
//         for (int j = 0; j < 16; ++j) {
//             sum += model.l3_weights[i][j] * l2_out[j];
//         }
//         l3_out[i] = std::clamp(sum >> 7, 0, 128); // Bring back to 128-scale
//     }

//     // --- Output: 32 -> 1 ---
//     int32_t final_sum = model.out_bias; // Already scaled by 128*128
//     for (int i = 0; i < 32; ++i) {
//         final_sum += model.out_weights[i] * l3_out[i];
//     }

//     // Final Scaling:
//     // Since we multiplied (Weight * Input) where both are scaled by 128, 
//     // and we did this across layers, the final_sum is effectively scaled by 128^3.
//     return static_cast<float>(final_sum) / (128.0f * 128.0f );
// }



void run_smoking_gun(const std::string& torch_model_path, const std::string& nnue_bin_path)
{
    auto state = rl::games::MigoyugoState::initialize();
    // 1. Load LibTorch Model
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state->get_observation_shape(), state->get_n_actions(),
        128, 512, 5, true);

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    network_ptr->load(torch_model_path);
    network_ptr->to(device);

    auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state->get_n_actions(), state->get_observation_shape());
    ev_ptr->evaluate(state);

    // 2. Load C++ NNUE Model
    NNUEModel nnue_model;
    FILE* f = fopen(nnue_bin_path.c_str(), "rb");
    fread(&nnue_model, sizeof(NNUEModel), 1, f);
    fclose(f);

    auto observation = state->get_observation();
    int active_count = 0;
    int16_t first_nonzero_acc = 0;

    for (size_t i = 0; i < observation.size(); i++) {
        if (observation[i] > 0.5f) {
            active_count++;
            // Check if the loop is actually adding anything
            first_nonzero_acc = nnue_model.l1_weights[0][i];
        }
    }

    // std::cout << "MOVE 1 DEBUG:" << std::endl;
    // std::cout << "Active Pieces found: " << active_count << std::endl;
    // std::cout << "First Weight Sample (Weight[0][" << (active_count > 0 ? "some_index" : "N/A") << "]): " << first_nonzero_acc << std::endl;
}

void run_sanity_check(const std::string& torch_model_path, const std::string& nnue_bin_path, const std::string& nnue_torch_path) {
    auto state = rl::games::MigoyugoState::initialize();

    torch::jit::script::Module torch_model;
    try {
        torch_model = torch::jit::load(nnue_torch_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the torch model\n";
        return;
    }


    // 1. Load LibTorch Model
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state->get_observation_shape(), state->get_n_actions(),
        128, 512, 5, true);

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    network_ptr->load(torch_model_path);
    network_ptr->to(device);

    auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state->get_n_actions(), state->get_observation_shape());
    ev_ptr->evaluate(state);

    // 2. Load C++ NNUE Model
    NNUEModel nnue_model;
    FILE* f = fopen(nnue_bin_path.c_str(), "rb");
    fread(&nnue_model, sizeof(NNUEModel), 1, f);
    fclose(f);


    std::cout << std::left << std::setw(10) << "Move"
        << std::setw(15) << "Torch Eval"
        << std::setw(15) << "NNUE Eval"
        << "Difference" << std::endl;
    std::cout << std::string(50, '-') << std::endl;



    for (int i = 0; i < 100 && !state->is_terminal(); ++i) {
        // --- A. Get Torch Eval ---
        // Convert state to tensor {1, channels, rows, cols}
        auto [probs, output] = ev_ptr->evaluate(state);
        float torch_val = output[0];


        // --- LIBTORCH EVAL ---
        auto obs = state->get_observation();
        torch::Tensor input = torch::from_blob(obs.data(), { 1, 256 }).clone();
        at::Tensor tensor_output = torch_model.forward({ input }).toTensor();
        float torch_eval = tensor_output.item<float>();




        // --- B. Get NNUE Eval ---
        // We assume your state has a method to get the sparse features for the accumulator
        int16_t accumulator[256] = { 0 };
        // Manually fill the accumulator for this check (L1 Bias + Weights)
        // In a real engine, this is updated incrementally
        for (int j = 0; j < 256; ++j) {
            accumulator[j] = nnue_model.l1_bias[j];
        }

        // 2. Add Piece Weights
        auto observation = state->get_observation();
        int active_count = 0;
        for (size_t i = 0; i < observation.size(); i++) {
            if (observation[i] > 0.5f) {
                active_count++;
                for (int j = 0; j < 256; ++j) {
                    accumulator[j] += nnue_model.l1_weights[j][i];
                }
            }
        }

        // std::cout << "MOVE " << i << "DEBUG:" << std::endl;
        // std::cout << "Active Pieces found: " << active_count << std::endl;
        // std::cout << "First Weight Sample (Weight[0][" << (active_count > 0 ? "some_index" : "N/A") << "]): " << std::endl;

        float nnue_val = evaluate_nnue(accumulator, nnue_model);

        // --- C. Compare ---
        float diff = std::abs(torch_eval - nnue_val);
        std::cout << std::left << std::setw(10) << i
            << std::setw(15) << torch_eval
            << std::setw(15) << nnue_val
            << (diff > 0.05 ? "⚠️ " : "✅ ") << diff << std::endl;

        // Make a random move to progress the game
        auto moves = state->actions_mask();
        auto action = choose_action(probs, state->get_n_actions());
        // auto action = choose_action(moves, state->get_n_actions());
        state = state->step(action);
    }
}


void run_sanity_check2(const std::string& torch_model_path, const std::string& nnue_bin_path, const std::string& nnue_torch_path) {
    auto state = rl::games::MigoyugoState::initialize();

    torch::jit::script::Module torch_model;
    try {
        torch_model = torch::jit::load(nnue_torch_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the torch model\n";
        return;
    }


    // 1. Load LibTorch Model
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state->get_observation_shape(), state->get_n_actions(),
        128, 512, 5, true);

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    network_ptr->load(torch_model_path);
    network_ptr->to(device);

    auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state->get_n_actions(), state->get_observation_shape());
    ev_ptr->evaluate(state);

    // 2. Load C++ NNUE Model
    NNUEModel nnue_model;
    FILE* f = fopen(nnue_bin_path.c_str(), "rb");
    fread(&nnue_model, sizeof(NNUEModel), 1, f);
    fclose(f);


    std::cout << std::left << std::setw(10) << "Move"
        << std::setw(15) << "Torch Eval"
        << std::setw(15) << "NNUE Eval"
        << "Difference" << std::endl;
    std::cout << std::string(50, '-') << std::endl;



    for (int i = 0; i < 100 && !state->is_terminal(); ++i) {
        // --- A. Get Torch Eval ---
        // Convert state to tensor {1, channels, rows, cols}
        auto [probs, output] = ev_ptr->evaluate(state);
        float torch_val = output[0];


        // --- LIBTORCH EVAL ---
        auto obs = state->get_observation();
        torch::Tensor input = torch::from_blob(obs.data(), { 1, 256 }).clone();
        at::Tensor tensor_output = torch_model.forward({ input }).toTensor();
        float torch_eval = tensor_output.item<float>();




        // --- B. Get NNUE Eval ---
        // We assume your state has a method to get the sparse features for the accumulator
        // int16_t accumulator[256] = { 0 };
        std::array<int16_t, 256> accumulator{};
        // Manually fill the accumulator for this check (L1 Bias + Weights)
        // In a real engine, this is updated incrementally
        for (int j = 0; j < 256; ++j) {
            accumulator[j] = nnue_model.l1_bias[j];
        }

        // 2. Add Piece Weights
        auto observation = state->get_observation();
        int active_count = 0;
        for (size_t i = 0; i < observation.size(); i++) {
            if (observation[i] > 0.5f) {
                active_count++;
                for (int j = 0; j < 256; ++j) {
                    accumulator[j] += nnue_model.l1_weights[j][i];
                }
            }
        }

        // std::cout << "MOVE " << i << "DEBUG:" << std::endl;
        // std::cout << "Active Pieces found: " << active_count << std::endl;
        // std::cout << "First Weight Sample (Weight[0][" << (active_count > 0 ? "some_index" : "N/A") << "]): " << std::endl;

        float nnue_val = evaluate_nnue2(accumulator, nnue_model);

        // --- C. Compare ---
        float diff = std::abs(torch_eval - nnue_val);
        std::cout << std::left << std::setw(10) << i
            << std::setw(15) << torch_eval
            << std::setw(15) << nnue_val
            << (diff > 0.05 ? "⚠️ " : "✅ ") << diff << std::endl;

        // Make a random move to progress the game
        auto moves = state->actions_mask();
        auto action = choose_action(probs, state->get_n_actions());
        // auto action = choose_action(moves, state->get_n_actions());
        state = state->step(action);
    }
}


void run_sanity_check_simd(const std::string& torch_model_path, const std::string& nnue_bin_path, const std::string& nnue_torch_path) {
    auto state = rl::games::MigoyugoState::initialize();
    auto lstate = rl::games::MigoyugoLightState::initialize_state();
    torch::jit::script::Module torch_model;
    try {
        torch_model = torch::jit::load(nnue_torch_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the torch model\n";
        return;
    }


    // 1. Load LibTorch Model
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state->get_observation_shape(), state->get_n_actions(),
        128, 512, 5, true);

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    network_ptr->load(torch_model_path);
    network_ptr->to(device);

    auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state->get_n_actions(), state->get_observation_shape());
    ev_ptr->evaluate(state);

    // 2. Load C++ NNUE Model
    NNUEModel nnue_model;
    FILE* f = fopen(nnue_bin_path.c_str(), "rb");
    fread(&nnue_model, sizeof(NNUEModel), 1, f);
    fclose(f);


    std::cout << std::left << std::setw(10) << "Move"
        << std::setw(15) << "Torch Eval"
        << std::setw(15) << "NNUE Eval"
        << "Difference" << std::endl;
    std::cout << std::string(50, '-') << std::endl;



    for (int i = 0; i < 100 && !state->is_terminal(); ++i) {
        // --- A. Get Torch Eval ---
        // Convert state to tensor {1, channels, rows, cols}
        auto [probs, output] = ev_ptr->evaluate(state);
        float torch_val = output[0];


        // --- LIBTORCH EVAL ---
        auto obs = state->get_observation();
        torch::Tensor input = torch::from_blob(obs.data(), { 1, 256 }).clone();
        at::Tensor tensor_output = torch_model.forward({ input }).toTensor();
        float torch_eval = tensor_output.item<float>();




        // --- B. Get NNUE Eval ---
        // We assume your state has a method to get the sparse features for the accumulator
        // int16_t accumulator[256] = { 0 };
        std::array<int16_t, 256> accumulator{};
        // Manually fill the accumulator for this check (L1 Bias + Weights)
        // In a real engine, this is updated incrementally
        for (int j = 0; j < 256; ++j) {
            accumulator[j] = nnue_model.l1_bias[j];
        }

        // 2. Add Piece Weights
        auto observation = state->get_observation();
        int active_count = 0;
        for (size_t i = 0; i < observation.size(); i++) {
            if (observation[i] > 0.5f) {
                active_count++;
                for (int j = 0; j < 256; ++j) {
                    accumulator[j] += nnue_model.l1_weights[j][i];
                }
            }
        }

        // std::cout << "MOVE " << i << "DEBUG:" << std::endl;
        // std::cout << "Active Pieces found: " << active_count << std::endl;
        // std::cout << "First Weight Sample (Weight[0][" << (active_count > 0 ? "some_index" : "N/A") << "]): " << std::endl;

        float nnue_val = evaluate_nnue_simd(accumulator, nnue_model);

        // --- C. Compare ---
        float diff = std::abs(torch_eval - nnue_val);
        std::cout << std::left << std::setw(10) << i
            << std::setw(15) << torch_eval
            << std::setw(15) << nnue_val
            << (diff > 0.05 ? "⚠️ " : "✅ ") << diff << std::endl;

        // Make a random move to progress the game
        auto moves = state->actions_mask();
        auto action = choose_action(probs, state->get_n_actions());
        // auto action = choose_action(moves, state->get_n_actions());
        state = state->step(action);
    }
}


void run_sanity_check_simd2(const std::string& torch_model_path, const std::string& nnue_bin_path, const std::string& nnue_torch_path) {
    auto state = rl::games::MigoyugoState::initialize();
    auto lstate = rl::games::MigoyugoLightState::initialize_state();
    torch::jit::script::Module torch_model;
    try {
        torch_model = torch::jit::load(nnue_torch_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the torch model\n";
        return;
    }


    // 1. Load LibTorch Model
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state->get_observation_shape(), state->get_n_actions(),
        128, 512, 5, true);

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    network_ptr->load(torch_model_path);
    network_ptr->to(device);

    auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state->get_n_actions(), state->get_observation_shape());
    ev_ptr->evaluate(state);

    // 2. Load C++ NNUE Model
    NNUEModel nnue_model;
    FILE* f = fopen(nnue_bin_path.c_str(), "rb");
    fread(&nnue_model, sizeof(NNUEModel), 1, f);
    fclose(f);


    std::cout << std::left << std::setw(10) << "Move"
        << std::setw(15) << "Torch Eval"
        << std::setw(15) << "NNUE Eval"
        << "Difference" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::array<int16_t, 256> white_accumulator{};
    std::array<int16_t, 256> black_accumulator{};
    // Manually fill the accumulator for this check (L1 Bias + Weights)
    // In a real engine, this is updated incrementally
    for (int j = 0; j < 256; ++j) {
        white_accumulator[j] = nnue_model.l1_bias[j];
        black_accumulator[j] = nnue_model.l1_bias[j];
    }


    rl::games::NNUEUpdate update{};
    // no active features at the start

    for (int i = 0; i < 100 && !state->is_terminal(); ++i) {
        // --- A. Get Torch Eval ---
        // Convert state to tensor {1, channels, rows, cols}
        auto [probs, output] = ev_ptr->evaluate(state);
        float torch_val = output[0];


        // --- LIBTORCH EVAL ---
        auto obs = state->get_observation();
        torch::Tensor input = torch::from_blob(obs.data(), { 1, 256 }).clone();
        at::Tensor tensor_output = torch_model.forward({ input }).toTensor();
        float torch_eval = tensor_output.item<float>();




        // --- B. Get NNUE Eval ---
        // We assume your state has a method to get the sparse features for the accumulator
        // int16_t accumulator[256] = { 0 };
        std::array<int16_t, 256> accumulator{};
        // Manually fill the accumulator for this check (L1 Bias + Weights)
        // In a real engine, this is updated incrementally
        for (int j = 0; j < 256; ++j) {
            accumulator[j] = nnue_model.l1_bias[j];
        }

        // 2. Add Piece Weights
        auto observation = state->get_observation();
        int active_count = 0;
        for (size_t i = 0; i < observation.size(); i++) {
            if (observation[i] > 0.5f) {
                active_count++;
                for (int j = 0; j < 256; ++j) {
                    accumulator[j] += nnue_model.l1_weights[j][i];
                }
            }
        }

        for (auto feat : update.white_added)
        {
            for (int j = 0;j < 256;j++)
            {
                white_accumulator[j] += nnue_model.l1_weights[j][feat];
            }
        }
        for (auto feat : update.black_added)
        {
            for (int j = 0;j < 256;j++)
            {
                black_accumulator[j] += nnue_model.l1_weights[j][feat];
            }
        }


        for (auto feat : update.white_removed)
        {
            for (int j = 0;j < 256;j++)
            {
                white_accumulator[j] -= nnue_model.l1_weights[j][feat];
            }
        }
        for (auto feat : update.black_removed)
        {
            for (int j = 0;j < 256;j++)
            {
                black_accumulator[j] -= nnue_model.l1_weights[j][feat];
            }
        }

        // std::cout << "MOVE " << i << "DEBUG:" << std::endl;
        // std::cout << "Active Pieces found: " << active_count << std::endl;
        // std::cout << "First Weight Sample (Weight[0][" << (active_count > 0 ? "some_index" : "N/A") << "]): " << std::endl;

        float nnue_val = evaluate_nnue_simd(accumulator, nnue_model);

        float cumulative_nnue_val = 0;

        if (lstate->player_turn() == 0)
        {
            cumulative_nnue_val = evaluate_nnue_simd(white_accumulator, nnue_model);
        }
        else {
            cumulative_nnue_val = evaluate_nnue_simd(black_accumulator, nnue_model);
        }

        // --- C. Compare ---
        float diff = std::abs(nnue_val - cumulative_nnue_val);
        std::cout << std::left << std::setw(10) << i
            << std::setw(15) << nnue_val
            << std::setw(15) << cumulative_nnue_val
            << (diff > 0.05 ? "⚠️ " : "✅ ") << diff << std::endl;

        // Make a random move to progress the game
        auto moves = state->actions_mask();
        auto action = choose_action(probs, state->get_n_actions());
        // auto action = choose_action(moves, state->get_n_actions());
        update.white_added.clear();
        update.white_removed.clear();
        update.black_added.clear();
        update.black_removed.clear();
        state = state->step(action);
        lstate = lstate->step_state_light(action, update);
    }
}

int main() {
    const std::string folder_name = "../checkpoints";
    std::filesystem::path folder(folder_name);
    std::filesystem::path nn_path;
    nn_path = folder / "migoyugo_strongest_900.pt";

    std::filesystem::path nnue_path;
    nnue_path = folder / "nnue_weights.bin";

    std::filesystem::path nnue_pt_path;
    nnue_pt_path = folder / "nnue_traced.pt";
    // run_sanity_check(nn_path.string(), nnue_path.string(), nnue_pt_path.string());
    // run_sanity_check2(nn_path.string(), nnue_path.string(), nnue_pt_path.string());
    run_sanity_check_simd(nn_path.string(), nnue_path.string(), nnue_pt_path.string());
    // run_sanity_check_simd2(nn_path.string(), nnue_path.string(), nnue_pt_path.string());
    // run_smoking_gun(nn_path.string(), nnue_path.string());

}