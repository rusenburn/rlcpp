#include <common/exceptions.hpp>

namespace rl::common
{
    IllegalActionException::IllegalActionException(const std::string &error)
        : std::runtime_error{error}
    {
    }

    SteppingTerminalStateException::SteppingTerminalStateException(const std::string &error)
        : std::runtime_error{error}
    {
    }

    UnreachableCodeException::UnreachableCodeException(const std::string &error)
        : std::runtime_error{error}
    {
    }
} // namespace rl::common
