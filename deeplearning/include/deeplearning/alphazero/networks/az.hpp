#ifndef RL_DEEPLEARNING_ALPHAZERO_AZ_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_AZ_HPP_

#include <torch/torch.h>
#include <utility>
namespace rl::deeplearning::alphazero
{
class IAlphazeroNetwork
{
public:
    virtual ~IAlphazeroNetwork();
    virtual torch::autograd::variable_list parameters() = 0;
    virtual std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state) = 0;
    virtual void to(torch::DeviceType device) = 0;
    virtual torch::DeviceType device() = 0;
    virtual std::unique_ptr<IAlphazeroNetwork> deepcopy() = 0;
    virtual std::unique_ptr<IAlphazeroNetwork> copy() = 0;
    virtual void save(std::string file_path) = 0;
    virtual void load(std::string file_path) = 0;
};

template <typename SELF, typename T>
class AlphazeroNetwork : public IAlphazeroNetwork
{
protected:
    T mod_{ nullptr };
    torch::DeviceType dev_;

    virtual ~AlphazeroNetwork() override {};
    void deepcopyto(std::unique_ptr<SELF>& other_network)
    {
        std::string data;
        {
            std::ostringstream oss;
            torch::serialize::OutputArchive archive;
            mod_->save(archive);
            archive.save_to(oss);
            data = oss.str();
        }
        {
            std::istringstream iss(data);
            torch::serialize::InputArchive archive;
            archive.load_from(iss);
            other_network->mod_->load(archive);
        }
        other_network->to(dev_);
    }

public:
    AlphazeroNetwork(T mod,
        torch::DeviceType dev)
        : mod_{ mod }, dev_{ dev }
    {
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state) override
    {
        return mod_->forward(state);
    }
    torch::autograd::variable_list parameters() override
    {
        return mod_->parameters();
    }
    void to(torch::DeviceType device) override
    {
        mod_->to(device);
        dev_ = device;
    }
    torch::DeviceType device() override
    {
        return dev_;
    }

    void save(std::string file_path) override
    {
        torch::save(mod_, file_path);
    }
    void load(std::string file_path) override
    {
        torch::serialize::InputArchive arv;
        arv.load_from(file_path);
        mod_->load(arv);
        mod_->to(dev_);
    }
};
}

#endif