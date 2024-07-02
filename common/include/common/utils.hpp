#ifndef RL_COMMON_UTILS_HPP_
#define RL_COMMON_UTILS_HPP_

#include <vector>
namespace rl::common::utils
{
    void normalize_vector(const std::vector<float> &vec_out);
    std::vector<float> apply_temperature(const std::vector<float> &probs, float temperature);
} // namespace rl::common::utils

#endif