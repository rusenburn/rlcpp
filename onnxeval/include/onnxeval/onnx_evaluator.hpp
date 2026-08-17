#ifndef RL_ONNXEVAL_ONNX_EVALUATOR_HPP_
#define RL_ONNXEVAL_ONNX_EVALUATOR_HPP_

#include <array>
#include <memory>
#include <vector>
#include <players/evaluator.hpp>
#include <onnxeval/onnx_session.hpp>

namespace rl::onnxeval
{
/// @brief NetworkEvaluator's post-processing, without libtorch.
///
/// Everything after the forward pass is a deliberate line-for-line port of
/// rl::deeplearning::NetworkEvaluator: mask the policy with actions_mask(),
/// renormalize by the row sum, and reduce wdl to a scalar as w - l. Keeping the
/// arithmetic identical is what lets run/onnx_parity diff the two evaluators and
/// attribute any difference to the network backend rather than to the wrapper.
class OnnxEvaluator : public rl::players::IEvaluator
{
private:
    std::shared_ptr<IOnnxSession> session_ptr_;
    int n_actions_;
    std::array<int, 3> observation_shape_;

    // Scratch, reused across calls. Members rather than locals because a search
    // calls evaluate() thousands of times at a near-constant batch size, so the
    // vectors reach their high-water mark once and stop allocating.
    std::vector<float> observations_;
    std::vector<float> actions_mask_;
    std::vector<float> raw_probs_;
    std::vector<float> wdls_;

    void evaluate(const std::vector<const rl::common::IState*>& state_ptrs_vec,
        std::vector<float>& probs_out,
        std::vector<float>& values_out);

public:
    OnnxEvaluator(std::shared_ptr<IOnnxSession> session_ptr,
        int n_actions,
        const std::array<int, 3>& observation_shape);
    ~OnnxEvaluator() override;

    std::tuple<std::vector<float>, std::vector<float>> evaluate(const std::vector<const rl::common::IState*>& state_ptrs) override;
    std::tuple<std::vector<float>, std::vector<float>> evaluate(const rl::common::IState* state_ptrs) override;
    std::tuple<std::vector<float>, std::vector<float>> evaluate(const std::unique_ptr<rl::common::IState>& state_ptrs) override;

    /// @brief Both share the session; only the scratch buffers are per-instance.
    ///
    /// NetworkEvaluator distinguishes the two because a torch module owns mutable
    /// device state. An ONNX session is immutable once loaded and its Run is
    /// thread-safe, so an independent copy would buy nothing and cost another
    /// 40 MB of weights per search thread.
    std::unique_ptr<rl::players::IEvaluator> clone() const override;
    std::unique_ptr<rl::players::IEvaluator> copy() const override;
};

} // namespace rl::onnxeval

#endif
