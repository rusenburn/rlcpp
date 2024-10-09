#include <iostream>
#include <deeplearning/network_loader.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <games/games.hpp>
#include <memory>
#include <filesystem>
#include <sstream>
int main(int argc, char const *argv[])
{
    char inp_c[200];
    
    std::cin.getline(inp_c,200);
    std::string inp{inp_c};
    if (inp != "Y" && inp != "y") return 0;
    auto state_ptr = rl::games::SantoriniState::initialize();
    auto net_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(
        state_ptr->get_observation_shape(),
        state_ptr->get_n_actions(),
        128,
        512,
        5);
    const std::string folder_name = "../checkpoints";
    std::filesystem::path folder(folder_name);
    std::filesystem::path file_path;
    file_path = folder / "test_save.pt";
    net_ptr->save_full(file_path.string());
    try
    {
        auto new_net = rl::deeplearning::alphazero::load_network(file_path.string());
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }catch(...)
    {
        std::cout << "error was caught" << '\n';
        throw;
    }
}
