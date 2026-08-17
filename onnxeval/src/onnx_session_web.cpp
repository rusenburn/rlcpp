// onnxruntime-web backend. Only exists under Emscripten; the native build uses
// onnx_session_ort.cpp instead.
#if defined(__EMSCRIPTEN__)

#include <algorithm>
#include <emscripten.h>
#include <onnxeval/onnx_session_web.hpp>

namespace rl::onnxeval
{
namespace
{
// Pointers cross as int (wasm32), and the JS side indexes HEAPF32, so every
// pointer is shifted to a float index. HEAPF32 is re-read on each call rather
// than captured: ALLOW_MEMORY_GROWTH detaches the old view when the heap grows,
// and a search allocates, so a cached view here would be a use-after-detach.
//
// Module.azRun is installed by az_worker.js and returns a Promise. Awaiting it
// is what suspends the wasm stack, which is why this module needs -sASYNCIFY.
EM_ASYNC_JS(int, az_run_js, (int obs_ptr, int n_states, int probs_ptr, int wdl_ptr), {
    // Module.azRun under MODULARIZE, self.azRun as a fallback so the bridge can
    // also be installed on the worker global. Whichever the worker set, one of
    // these finds it.
    const azRun = (typeof Module !== 'undefined' && Module.azRun) || self.azRun;
    if (!azRun) {
        return 1; // no session installed - mgy_az_init was never completed
    }
    try {
        await azRun(obs_ptr, n_states, probs_ptr, wdl_ptr);
        return 0;
    } catch (e) {
        console.error('azRun failed', e);
        return 2;
    }
});
} // namespace

WebOnnxSession::WebOnnxSession(const std::array<int, 3>& observation_shape, int n_actions, int fixed_batch)
    : observation_shape_{ observation_shape },
    n_actions_{ n_actions },
    fixed_batch_{ fixed_batch > 0 ? fixed_batch : 1 },
    observation_size_{ observation_shape.at(0) * observation_shape.at(1) * observation_shape.at(2) }
{
    padded_observations_.assign(static_cast<size_t>(fixed_batch_) * observation_size_, 0.0f);
    padded_probs_.assign(static_cast<size_t>(fixed_batch_) * n_actions_, 0.0f);
    padded_wdl_.assign(static_cast<size_t>(fixed_batch_) * 3, 0.0f);
}

WebOnnxSession::~WebOnnxSession() = default;

void WebOnnxSession::run(const float* observations, int n_states, float* probs_out, float* wdl_out)
{
    if (n_states <= 0) return;

    // A batch bigger than the configured one would be a new shape too, so grow
    // the pad to match and keep using that size from then on. In practice this
    // fires at most once: Amcts2 never collects more than max_async_simulations.
    if (n_states > fixed_batch_)
    {
        fixed_batch_ = n_states;
        padded_observations_.assign(static_cast<size_t>(fixed_batch_) * observation_size_, 0.0f);
        padded_probs_.assign(static_cast<size_t>(fixed_batch_) * n_actions_, 0.0f);
        padded_wdl_.assign(static_cast<size_t>(fixed_batch_) * 3, 0.0f);
    }

    const bool needs_padding = n_states < fixed_batch_;
    const float* run_in = observations;
    float* run_probs = probs_out;
    float* run_wdl = wdl_out;

    if (needs_padding)
    {
        // Only the live rows are copied; the tail keeps whatever it held. The
        // padding rows are evaluated and thrown away, so their contents are
        // irrelevant as long as they are finite - a zeroed board is.
        std::copy(observations,
            observations + static_cast<size_t>(n_states) * observation_size_,
            padded_observations_.begin());
        run_in = padded_observations_.data();
        run_probs = padded_probs_.data();
        run_wdl = padded_wdl_.data();
    }

    const int rc = az_run_js(reinterpret_cast<int>(run_in),
        fixed_batch_,
        reinterpret_cast<int>(run_probs),
        reinterpret_cast<int>(run_wdl));

    // Only the live rows count; the padding is an artifact of keeping one shape
    // for WebGPU and would inflate the reported speed.
    if (rc == 0) evaluations_ += n_states;

    if (rc == 0 && needs_padding)
    {
        std::copy(padded_probs_.begin(),
            padded_probs_.begin() + static_cast<size_t>(n_states) * n_actions_,
            probs_out);
        std::copy(padded_wdl_.begin(),
            padded_wdl_.begin() + static_cast<size_t>(n_states) * 3,
            wdl_out);
    }

    if (rc != 0)
    {
        // This module is built without exception support, like migoyugo_wasm, so
        // a failure is reported by leaving a uniform, finite result rather than
        // by throwing. The search then behaves as if the position were unknown
        // instead of the whole module aborting.
        const float uniform = n_actions_ > 0 ? 1.0f / static_cast<float>(n_actions_) : 0.0f;
        for (int i = 0; i < n_states; i++)
        {
            for (int j = 0; j < n_actions_; j++)
            {
                probs_out[i * n_actions_ + j] = uniform;
            }
            wdl_out[i * 3 + 0] = 0.0f;
            wdl_out[i * 3 + 1] = 1.0f;
            wdl_out[i * 3 + 2] = 0.0f;
        }
    }
}

} // namespace rl::onnxeval

#endif // __EMSCRIPTEN__
