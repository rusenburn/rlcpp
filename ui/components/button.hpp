#ifndef RL_UI_BUTTON
#define RL_UI_BUTTON

#include "component.hpp"
#include <raylib.h>
#include <string>
#include <common/observer.hpp>

namespace rl::ui
{
class Button : public IComponent, public common::Subject<int> {
public:
    Button(std::string text, Rectangle rect, Color color);
    ~Button();
    void handle_events()override;
    void draw()override;

private:
    std::string text_;
    Rectangle rect_;
    Color color_;
};
} // namespace rl::ui



#endif