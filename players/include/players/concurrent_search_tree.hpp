#ifndef RL_PLAYERS_CONCURRENT_SEARCH_TREE_HPP_
#define RL_PLAYERS_CONCURRENT_SEARCH_TREE_HPP_

#include "search_tree.hpp"
#include <vector>
namespace rl::players
{
class IConcurrentSearchTree : public ISearchTree
{
    public:
    virtual ~IConcurrentSearchTree()override;
    virtual std::pair<std::vector<std::vector<float>>,std::vector<float>> search_multiple(const std::vector<const rl::common::IState*>& state_ptrs, int minimum_sims,std::chrono::duration<int, std::milli> minimum_duration) = 0;
};
} // namespace rl::players


#endif
