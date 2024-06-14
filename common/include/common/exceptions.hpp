#ifndef RL_COMMON_EXCEPTIONS_HPP_
#define RL_COMMON_EXCEPTIONS_HPP_

#include <stdexcept>
#include <string>
namespace rl::common
{
    class IllegalActionException : public std::runtime_error
    {
    public:
        IllegalActionException(const std::string &error);
    };

    class SteppingTerminalStateException : public std::runtime_error
    {
    public:
        SteppingTerminalStateException(const std::string &error);
        
    };

    class UnreachableCodeException : public std::runtime_error
    {
    public:
        UnreachableCodeException(const std::string &error);
        
    };
} // namespace rl::common

#endif
