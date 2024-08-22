#include <players/search_tree.hpp>


namespace rl::players
{
ISearchTree::~ISearchTree() = default;


std::vector<float> ISearchTree::search(const rl::common::IState* state_ptr, int minimum_no_simulations)
{
    return search(state_ptr, minimum_no_simulations, std::chrono::duration<int, std::milli>(0));
};
std::vector<float> ISearchTree::search(const rl::common::IState* state_ptr, std::chrono::duration<int, std::milli> duration)
{
    return search(state_ptr, 2, duration);
};
} // namespace rl::common
