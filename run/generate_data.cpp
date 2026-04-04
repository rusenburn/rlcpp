#include <memory>
#include <iostream>
#include <string>

#include "nnue_data_generator.hpp"
#include <deeplearning/alphazero/networks/shared_res_nn.hpp> // Your specific network header
#include <games/migoyugo.hpp>               // Your specific game state header

int main() {
    // 1. Setup State and Shapes
    auto state_ptr = rl::games::MigoyugoState::initialize();
    auto obs_shape = state_ptr->get_observation_shape();
    int n_actions = state_ptr->get_n_actions();

    // 2. Setup Network
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(
        obs_shape, n_actions, 128, 512, 5, true);
    
    // Logic for loading checkpoints...
    const std::string folder_name = "../checkpoints";
    std::string load_name = "migoyugo_strongest_900.pt"; // Ensure this isn't empty!
    std::filesystem::path file_path = std::filesystem::path(folder_name) / load_name;
    
    network_ptr->load(file_path.string());
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    network_ptr->to(device);

    // 3. Setup Evaluator
    // Wrap the network in the evaluator
    auto evaluator = std::make_shared<rl::deeplearning::NetworkEvaluator>(
        std::move(network_ptr), 
        n_actions, 
        obs_shape
    );

    // 4. Initialize Generator
    rl::training::NNUEDataGenerator generator(evaluator, 1024);

    // 5. Run Generation
    int total_samples = 100000;
    std::string output_path = "training_data.bin";
    
    std::cout << "Starting generation..." << std::endl;
    
    try {
        // Use *state_ptr to pass the actual object by reference
        generator.generate(*state_ptr, total_samples, output_path, 1.0f);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}