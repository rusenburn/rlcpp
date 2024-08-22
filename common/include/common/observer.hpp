#ifndef RL_COMMON_OBSERVER_HPP_
#define RL_COMMON_OBSERVER_HPP_
#include <memory>
#include <functional>
#include <vector>
#include <list>

namespace rl::common
{

template <typename T>
class Observer
{
private:
    std::function<void(T)> fn_{ nullptr };

public:
    Observer(std::function<void(T)> fn)
        : fn_(fn)
    {
    }
    ~Observer() = default;
    void inoke(T a)
    {
        if (fn_ != nullptr)
        {
            fn_(a);
        }
    }
    void unsubscribe()
    {
        if (fn_ != nullptr)
        {
            fn_ = nullptr;
        }
    }
    void resub(std::function<void(T)> fn)
    {
        fn_ = fn;
    }
};

template <typename Type>
class Subject
{
private:
    std::list<std::weak_ptr<Observer<Type>>> observers_;

public:
    Subject()
        : observers_{}
    {
    }
    ~Subject()
    {
        observers_.clear();
    }
    std::shared_ptr<Observer<Type>> subscribe(std::function<void(Type)>& fn)
    {
        std::shared_ptr<Observer<Type>> a = std::make_shared<Observer<Type>>(fn);
        std::weak_ptr<Observer<Type>> weak;
        weak = a;
        observers_.push_back(weak);
        return a;
    }

    void notify(Type obj)
    {

        auto i = observers_.begin();
        while (i != observers_.end())
        {
            auto& obs_ptr = *i;
            if (obs_ptr.expired())
            {
                i = observers_.erase(i);
            }
            else
            {
                obs_ptr.lock()->inoke(obj);
                i++;
            }
        }
    }
};

}


#endif