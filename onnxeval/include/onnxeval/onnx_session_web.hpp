#ifndef RL_ONNXEVAL_ONNX_SESSION_WEB_HPP_
#define RL_ONNXEVAL_ONNX_SESSION_WEB_HPP_

#if defined(__EMSCRIPTEN__)

#include <array>
#include <vector>
#include <onnxeval/onnx_session.hpp>

namespace rl::onnxeval
{
/// @brief IOnnxSession backed by onnxruntime-web, so the network runs on WebGPU.
///
/// There is no C++ API for WebGPU inference - onnxruntime-web is JavaScript, and
/// its run() returns a Promise. This class crosses that boundary with
/// EM_ASYNC_JS, which requires the module to be linked with -sASYNCIFY.
///
/// Asyncify is affordable here specifically because of how Amcts2 is written.
/// Amcts2::search calls the evaluator at the top of its loop, *after* the tree
/// recursion in roll() has already returned, so the stack that has to be
/// suspended is only:
///
///     mgy_az_bot_move -> Amcts2::search -> OnnxEvaluator::evaluate -> run -> JS
///
/// The deep MCTS recursion is never live across a suspend, so Asyncify does not
/// have to instrument it.
///
/// The JS side is installed by wasm/web/az_worker.js as Module.azRun before any
/// search starts; see wasm/migoyugo_az_wasm.cpp for the init handshake.
class WebOnnxSession : public IOnnxSession
{
public:
    /// @param fixed_batch every run is padded up to this many rows; see run()
    WebOnnxSession(const std::array<int, 3>& observation_shape, int n_actions, int fixed_batch);
    ~WebOnnxSession() override;

    /// @brief Suspends the wasm stack until onnxruntime-web resolves.
    ///
    /// The batch is padded to a constant size before it crosses into JS, and the
    /// padding rows are dropped on the way back. This is not an optimization,
    /// it is what makes the WebGPU backend usable at all: ort-web compiles a
    /// fresh set of shaders for every distinct input shape it sees, and Amcts2
    /// hands over a variable number of leaves - roll() drops rollouts that come
    /// back null, and the flush after the search loop is whatever remains. Left
    /// alone that produces a new shape every few batches, and each one pays full
    /// shader compilation for all 13 convolutions and 14 gemms.
    void run(const float* observations, int n_states, float* probs_out, float* wdl_out) override;

    /// @brief Live positions evaluated since the last reset, padding excluded.
    ///
    /// Amcts2 reports no simulation count and its sources are off limits, so this
    /// is the honest stand-in for a speed readout: one leaf evaluation is roughly
    /// one simulation that reached a new node. Report it as evaluations per
    /// second, never as nodes per second - it is not comparable to the NNUE
    /// engine's node count, which counts every alpha-beta node visited.
    long long evaluations() const { return evaluations_; }
    void reset_evaluations() { evaluations_ = 0; }

private:
    std::array<int, 3> observation_shape_;
    int n_actions_;
    int fixed_batch_;
    int observation_size_;
    long long evaluations_{ 0 };

    std::vector<float> padded_observations_;
    std::vector<float> padded_probs_;
    std::vector<float> padded_wdl_;
};
} // namespace rl::onnxeval

#endif // __EMSCRIPTEN__

#endif
