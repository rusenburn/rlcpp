// Native ONNX Runtime backend. Excluded under Emscripten, where inference goes
// through onnxruntime-web instead and this API does not exist.
#if !defined(__EMSCRIPTEN__)

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <onnxeval/onnx_session_ort.hpp>

namespace rl::onnxeval
{
namespace
{
// One environment for the whole process. ONNX Runtime documents Env as
// something to create once and share; creating one per session leaks threads.
Ort::Env& shared_env()
{
    static Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "rlcpp" };
    return env;
}

std::string name_at(const Ort::Session& session, size_t index, bool input)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto held = input ? session.GetInputNameAllocated(index, allocator)
        : session.GetOutputNameAllocated(index, allocator);
    return std::string{ held.get() };
}

// A dimension is -1 when the exporter made it dynamic. Only the fixed ones are
// worth checking, and a mismatch there means the .onnx was exported for a
// different game.
void expect_dim(const std::vector<int64_t>& shape, size_t index, int expected,
    const char* what, const std::string& model_path)
{
    if (index >= shape.size() || shape.at(index) < 0 || shape.at(index) == expected)
    {
        return;
    }
    std::ostringstream oss;
    oss << model_path << ": expected " << what << " " << expected
        << " but the model declares " << shape.at(index);
    throw std::runtime_error(oss.str());
}
} // namespace

struct OrtOnnxSession::Impl
{
    Ort::SessionOptions options{};
    Ort::Session session{ nullptr };
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::string input_name;
    std::string probs_name;
    std::string wdl_name;
    std::string provider;

    std::array<int, 3> observation_shape{};
    int n_actions{ 0 };
};

OrtOnnxSession::OrtOnnxSession(const std::string& model_path,
    const std::array<int, 3>& observation_shape,
    int n_actions,
    OrtProvider provider,
    int intra_op_threads)
    : impl_{ std::make_unique<Impl>() }
{
    impl_->observation_shape = observation_shape;
    impl_->n_actions = n_actions;

    impl_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (intra_op_threads > 0)
    {
        impl_->options.SetIntraOpNumThreads(intra_op_threads);
    }

    impl_->provider = "CPU";
    if (provider == OrtProvider::cuda)
    {
        // A CPU-only ONNX Runtime distribution has no CUDA provider, and asking
        // for one throws. Falling back keeps a machine without the GPU package
        // working instead of failing at construction.
        try
        {
            OrtCUDAProviderOptions cuda_options{};
            impl_->options.AppendExecutionProvider_CUDA(cuda_options);
            impl_->provider = "CUDA";
        }
        catch (const Ort::Exception& e)
        {
            std::cout << "OrtOnnxSession: CUDA provider unavailable (" << e.what()
                << "); falling back to CPU\n";
        }
    }

#if defined(_WIN32)
    const std::wstring wide_path(model_path.begin(), model_path.end());
    impl_->session = Ort::Session{ shared_env(), wide_path.c_str(), impl_->options };
#else
    impl_->session = Ort::Session{ shared_env(), model_path.c_str(), impl_->options };
#endif

    if (impl_->session.GetInputCount() != 1 || impl_->session.GetOutputCount() != 2)
    {
        throw std::runtime_error(model_path + ": expected 1 input and 2 outputs (probs, wdl)");
    }

    impl_->input_name = name_at(impl_->session, 0, true);
    const std::string out0 = name_at(impl_->session, 0, false);
    const std::string out1 = name_at(impl_->session, 1, false);
    // export_az_onnx.py names them, but positional order is the export order
    // anyway, so an unnamed graph from another exporter still works.
    const bool swapped = (out0 == "wdl" || out1 == "probs");
    impl_->probs_name = swapped ? out1 : out0;
    impl_->wdl_name = swapped ? out0 : out1;

    const auto in_shape = impl_->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (in_shape.size() != 4)
    {
        throw std::runtime_error(model_path + ": input must be rank 4 (N,C,H,W)");
    }
    expect_dim(in_shape, 1, observation_shape.at(0), "input channels", model_path);
    expect_dim(in_shape, 2, observation_shape.at(1), "input height", model_path);
    expect_dim(in_shape, 3, observation_shape.at(2), "input width", model_path);

    const auto probs_shape = impl_->session.GetOutputTypeInfo(swapped ? 1 : 0)
        .GetTensorTypeAndShapeInfo().GetShape();
    expect_dim(probs_shape, 1, n_actions, "policy size", model_path);
}

OrtOnnxSession::~OrtOnnxSession() = default;

const std::string& OrtOnnxSession::provider_name() const
{
    return impl_->provider;
}

void OrtOnnxSession::run(const float* observations, int n_states, float* probs_out, float* wdl_out)
{
    const int64_t observation_size = static_cast<int64_t>(impl_->observation_shape.at(0))
        * impl_->observation_shape.at(1) * impl_->observation_shape.at(2);

    const std::array<int64_t, 4> in_shape{ n_states,
        impl_->observation_shape.at(0),
        impl_->observation_shape.at(1),
        impl_->observation_shape.at(2) };

    // CreateTensor wraps the caller's buffer without copying; ONNX Runtime does
    // not write to an input, so casting away const is safe here.
    Ort::Value input = Ort::Value::CreateTensor<float>(impl_->memory_info,
        const_cast<float*>(observations),
        static_cast<size_t>(n_states) * observation_size,
        in_shape.data(),
        in_shape.size());

    const char* input_names[] = { impl_->input_name.c_str() };
    const char* output_names[] = { impl_->probs_name.c_str(), impl_->wdl_name.c_str() };

    auto outputs = impl_->session.Run(Ort::RunOptions{ nullptr },
        input_names, &input, 1,
        output_names, 2);

    std::memcpy(probs_out, outputs.at(0).GetTensorData<float>(),
        sizeof(float) * static_cast<size_t>(n_states) * impl_->n_actions);
    std::memcpy(wdl_out, outputs.at(1).GetTensorData<float>(),
        sizeof(float) * static_cast<size_t>(n_states) * 3);
}

} // namespace rl::onnxeval

#endif // !__EMSCRIPTEN__
