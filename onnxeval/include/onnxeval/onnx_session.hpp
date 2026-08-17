#ifndef RL_ONNXEVAL_ONNX_SESSION_HPP_
#define RL_ONNXEVAL_ONNX_SESSION_HPP_

namespace rl::onnxeval
{
/// @brief The seam between OnnxEvaluator and whatever actually runs the graph.
///
/// Deliberately narrow - one call, raw pointers, no ONNX types. Natively this is
/// backed by ONNX Runtime (OrtOnnxSession). Under Emscripten it will be backed by
/// a call into onnxruntime-web, which is asynchronous and can therefore only be
/// implemented behind a bridge; keeping the interface this thin means that
/// backend can be added without OnnxEvaluator changing at all.
///
/// Implementations must be safe to call from multiple threads on one instance:
/// OnnxEvaluator::clone() and copy() share a session rather than duplicating a
/// multi-megabyte model, and the concurrent search trees evaluate in parallel.
class IOnnxSession
{
public:
    virtual ~IOnnxSession();

    /// @brief Runs the network over a batch.
    /// @param observations n_states * C*H*W floats, NCHW row-major
    /// @param n_states batch size
    /// @param probs_out receives n_states * n_actions policy values, unmasked
    /// @param wdl_out receives n_states * 3 win/draw/loss values
    virtual void run(const float* observations,
        int n_states,
        float* probs_out,
        float* wdl_out) = 0;
};
} // namespace rl::onnxeval

#endif
