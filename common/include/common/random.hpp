#ifndef RL_COMMON_RANDOM_MT_HPP_
#define RL_COMMON_RANDOM_MT_HPP_

#include <chrono>
#include <random>

namespace rl::common
{
    inline std::mt19937 generate()
    {
        std::random_device rd{};
        std::seed_seq ss{
            static_cast<std::seed_seq::result_type>(std::chrono::steady_clock::now().time_since_epoch().count()),
            rd(), rd(), rd(), rd(), rd(), rd(), rd()};

        return std::mt19937{ss};
    }

    inline std::mt19937 mt{generate()};

    
    /// @brief Generate random int number [min,max)
    /// @param min 
    /// @param max 
    /// @return 
    inline int get(int min, int max)
    {
        return std::uniform_int_distribution{min, max - 1}(mt);
    }

    /// @brief Generate random int number [0,max)
    /// @param max 
    /// @return 
    inline int get(int max)
    {
        return get(0, max);
    }

    /// @brief Generate random number [0,1)
    /// @return 
    inline float get()
    {
        return static_cast<float>(get(RAND_MAX)) / static_cast<float>(RAND_MAX);
    }
} // namespace rl::common

#endif
