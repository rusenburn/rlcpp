#ifndef RL_COMMON_STATE_HPP_
#define RL_COMMON_STATE_HPP_

#include <memory>
#include <vector>
#include <string>
namespace rl::common
{
    class IState
    {
    public:
        /// @brief 
        /// @param action the action performed by the player to transition the state to the next state
        /// @return a unique_ptr of the next state
        virtual std::unique_ptr<IState> step(int action) const = 0;


        /// @brief returns an initial state
        /// @return 
        virtual std::unique_ptr<IState> reset() const = 0;

        /// @brief renders the state
        virtual void render() const = 0;

        /// @brief checks if the state is terminal
        /// @return true is the state is terminal else returns false
        virtual bool is_terminal() const = 0;

        /// @brief returns a reward relative to the current player
        /// @return a float indicates the relative reward for the current player
        virtual float get_reward() const = 0;

        /// @brief 
        /// @return gets a copy of the observation for the state
        virtual std::vector<float> get_observation() const = 0;


        /// @brief short observation , used as a hashkey for maps
        /// @return a string
        virtual std::string to_short() const = 0;

        /// @brief 
        /// @return 
        virtual std::array<int,3> get_observation_shape() const = 0;

        virtual int get_n_actions() const = 0;

        virtual int player_turn() const = 0;

        virtual std::vector<bool> actions_mask() const = 0;

        /// @brief should return a deep copy for the current state , modifying the copy should not change the current state
        /// @return 
        virtual std::unique_ptr<IState> clone() const = 0;

        virtual ~IState();
    };

} // namespace rl::common

#endif