#ifndef RL_DEEPLEARNING_ALPHAZERO_NETWORK_NETWORK_TYPES_
#define RL_DEEPLEARNING_ALPHAZERO_NETWORK_NETWORK_TYPES_


namespace rl::deeplearning::alphazero
{
enum class NetworkType
{
    SmallAlpha = 1,
    SharedResidualNetwork = 2,
    TinyNetwork = 3

};
} // namespace rl::deeplearning::alphazero


#endif

