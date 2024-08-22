#include <common/utils.hpp>
#include <cmath>
#include <stdexcept>
#include <random>
namespace rl::common::utils
{
float EPS = 1e-6f;
void normalize_vector(std::vector<float>& vec)
{
    float sum = 0;
    for (auto p : vec)
    {
        sum += p;
    }

    float final_sum{ 0 };
    for (auto& p : vec)
    {
        p /= sum;
        final_sum += p;
    }
    if (final_sum < 1.0f - 1e-3 || final_sum > 1.0f + 1e-3f)
    {
        std::runtime_error("action probabilities do not equal to one");
    }
}
std::vector<float> apply_temperature(const std::vector<float>& probs_ref_out, float temperature)
{
    if (probs_ref_out.size() == 0)
    {
        return {};
    }
    if (temperature == 0)
    {
        float max{ -INFINITY };
        std::vector<int> maxes{};
        for (int i = 0; i < probs_ref_out.size(); i++)
        {
            if (probs_ref_out[i] > max)
            {
                max = probs_ref_out[i];
                maxes = { i };
            }
            else if (probs_ref_out[i] == max)
            {
                maxes.push_back(i);
            }
        }
        std::vector<float> result(probs_ref_out.size(), 0);
        int n_maxes = maxes.size();
        for (auto id : maxes)
        {
            result[id] = 1 / static_cast<float>(n_maxes);
        }
        return result;
    }

    float sum_probs{ 0 };
    std::vector<float> probs_with_temp{};
    probs_with_temp.reserve(probs_ref_out.size());
    for (auto p : probs_ref_out)
    {
        float p_t = powf(p, 1 / temperature);
        probs_with_temp.emplace_back(p_t);
    }

    normalize_vector(probs_with_temp);

    return probs_with_temp;
}

std::vector<float> rl::common::utils::get_dirichlet_noise(const int n_actions, const float alpha, std::mt19937& engine)
{
    std::vector<float> output;
    output.reserve(n_actions);
    std::gamma_distribution<> gamma(alpha, 1);
    for (int i = 0; i < n_actions; i++)
    {
        output.emplace_back(static_cast<float>(gamma(engine)));
    }
    normalize_vector(output);
    return output;
}

std::vector<float> get_dirichlet_noise(const std::vector<bool>& actions_mask, float alpha, std::mt19937& engine)
{
    const int n_actions = actions_mask.size();
    std::vector<float> output;
    output.reserve(n_actions);
    if (alpha < 0)
    {
        int sum_legal_actions = 0;
        for (bool a : actions_mask)
        {
            if (a)
            {
                sum_legal_actions += 1;
            }
        }
        alpha = 10 / sum_legal_actions;
    }
    std::gamma_distribution<> gamma(alpha, 1);

    for (int i = 0; i < n_actions; i++)
    {
        if (actions_mask.at(i))
        {
            output.emplace_back(static_cast<float>(gamma(engine)));
        }
        else
        {
            output.emplace_back(static_cast<float>(1e-8));
        }
    }
    normalize_vector(output);
    return output;
}

} // namespace rl::common::utils
