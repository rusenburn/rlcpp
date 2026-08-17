#ifndef RL_ONNXEVAL_ONNX_SESSION_ORT_HPP_
#define RL_ONNXEVAL_ONNX_SESSION_ORT_HPP_

#include <array>
#include <memory>
#include <string>
#include <onnxeval/onnx_session.hpp>

namespace rl::onnxeval
{
enum class OrtProvider
{
    cpu,
    cuda
};

/// @brief IOnnxSession backed by ONNX Runtime, for native builds.
///
/// The ONNX Runtime headers stay behind a pimpl so that consumers - run/
/// executables, ui/ - need neither the ORT include directory nor its C++ API in
/// their translation units. Only this library is compiled against it.
class OrtOnnxSession : public IOnnxSession
{
public:
    /// @param model_path a .onnx produced by scripts/export_az_onnx.py
    /// @param observation_shape {C,H,W}; validated against the model's input
    /// @param n_actions validated against the model's policy output
    /// @param provider cuda falls back to cpu with a message if unavailable
    /// @param intra_op_threads 0 leaves ONNX Runtime's default
    OrtOnnxSession(const std::string& model_path,
        const std::array<int, 3>& observation_shape,
        int n_actions,
        OrtProvider provider = OrtProvider::cpu,
        int intra_op_threads = 0);
    ~OrtOnnxSession() override;

    void run(const float* observations, int n_states, float* probs_out, float* wdl_out) override;

    /// @brief The provider actually in use, which may differ from the one asked for.
    const std::string& provider_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace rl::onnxeval

#endif
