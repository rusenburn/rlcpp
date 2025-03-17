#include "button.hpp"
#include <sstream>

namespace rl::ui
{
Button::Button(std::string text, Rectangle rect, Color color)
    :text_(text),
    rect_(rect), color_(color)

{
}

Button::~Button() = default;

void Button::handle_events()
{
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    {
        auto mouse_pos = GetMousePosition();
        bool is_button_pressed = CheckCollisionPointRec(mouse_pos, rect_);
        if (is_button_pressed)
        {
            notify(1);
        }
    }
}


void Button::draw()
{
    char text[200];
    strcpy(text, text_.c_str());
    DrawRectangleRec(rect_, color_);
    int initial_font_size = 16;
    int font_size = initial_font_size + 2;
    int text_width = 1000000;
    while (text_width > rect_.width / 2 || font_size + 4 > rect_.height)
    {
        font_size -= 2;
        text_width = MeasureText(text, font_size);
    }
    DrawText(text, rect_.x + rect_.width / 2 - text_width / 2, rect_.y + rect_.height / 2 - font_size / 2, font_size, WHITE);
}
} // namespace rl::ui

