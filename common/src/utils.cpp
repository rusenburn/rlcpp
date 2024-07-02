#include <common/utils.hpp>
#include <cmath>
#include <stdexcept>
namespace rl::common::utils
{
    float EPS = 1e-6f;
    void normalize_vector(std::vector<float> &vec)
    {
        float sum = 0;
        for (auto p : vec)
        {
            sum += p;
        }
        
        float final_sum{0};
        for (auto &p : vec)
        {
            p /= sum;
            final_sum += p;
        }
        if (final_sum < 1.0f - 1e-3 || final_sum > 1.0f + 1e-3f)
        {
            std::runtime_error("action probabilities do not equal to one");
        }
    }
    std::vector<float> apply_temperature(const std::vector<float> &probs_ref, float temperature)
    {
        if (probs_ref.size() == 0)
        {
            return {};
        }
        if (temperature == 0)
        {
            float max{-INFINITY};
            std::vector<int> maxes{};
            for (int i = 0; i < probs_ref.size(); i++)
            {
                if (probs_ref[i] > max)
                {
                    max = probs_ref[i];
                    maxes = {i};
                }
                else if (probs_ref[i] == max)
                {
                    maxes.push_back(i);
                }
            }
            std::vector<float> result(probs_ref.size(), 0);
            int n_maxes = maxes.size();
            for (auto id : maxes)
            {
                result[id] = 1 / static_cast<float>(n_maxes);
            }
            return result;
        }

        float sum_probs{0};
        std::vector<float> probs_with_temp{};
        probs_with_temp.reserve(probs_ref.size());
        for (auto p : probs_ref)
        {
            float p_t = powf(p, 1 / temperature);
            probs_with_temp.emplace_back(p_t);
        }

        normalize_vector(probs_with_temp);

        return probs_with_temp;
    }
} // namespace rl::common::utils
