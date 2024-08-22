#include <deeplearning/alphazero/networks/squeeze_and_excite.hpp>

namespace rl::deeplearning::alphazero
{
SqueezeAndExciteImpl::SqueezeAndExciteImpl(int channels, int squeeze_rate)
    : channels_{ channels },
    prepare_(torch::nn::Sequential{
        torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)) }),
        fcs_{ torch::nn::Sequential{
            torch::nn::Flatten(),
            torch::nn::Linear(torch::nn::LinearOptions(channels, channels / squeeze_rate)),
            torch::nn::ReLU(),
            torch::nn::Linear(torch::nn::LinearOptions(channels / squeeze_rate, channels * 2))} }
{
    register_module("prepare_", prepare_);
    register_module("fcs_", fcs_);
}

SqueezeAndExciteImpl::~SqueezeAndExciteImpl() = default;


torch::Tensor SqueezeAndExciteImpl::forward(torch::Tensor state, torch::Tensor input_)
{
    torch::Tensor prepared = prepare_->forward(state);
    prepared = fcs_->forward(prepared);
    auto splitted = prepared.split(channels_, 1);
    torch::Tensor w = splitted[0];
    torch::Tensor b = splitted[1];
    torch::Tensor z = w.sigmoid();
    z = z.unsqueeze(-1).unsqueeze(-1).expand_as(input_);
    b = b.unsqueeze(-1).unsqueeze(-1).expand_as(input_);
    torch::Tensor output = (state * z) + b;
    return output;
}
} // namespace rl::deeplearning::alphazero