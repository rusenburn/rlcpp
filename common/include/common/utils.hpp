#ifndef RL_COMMON_UTILS_HPP_
#define RL_COMMON_UTILS_HPP_

#include <vector>
#include <random>
namespace rl::common::utils
{
void normalize_vector(std::vector<float>& vec_out);
std::vector<float> apply_temperature(const std::vector<float>& probs_ref_out, float temperature);
std::vector<float> get_dirichlet_noise(const int n_action, const float alpha, std::mt19937& engine);
std::vector<float> get_dirichlet_noise(const std::vector<bool>& actions_mask, const float alpha, std::mt19937& engine);
} // namespace rl::common::utils

#endif