#ifndef RL_RUN_CONSOLE_HPP_
#define RL_RUN_CONSOLE_HPP_

#include <memory>

namespace rl::run
{

class IConsole
{
public:
    virtual ~IConsole();
    virtual void run() = 0;
};
} // namespace rl::run

#endif
