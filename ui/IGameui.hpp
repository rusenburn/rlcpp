#ifndef RL_UI_GAMEUI_HPP_
#define RL_UI_GAMEUI_HPP_

namespace rl::ui
{
    class IGameui
    {
    public:
        virtual ~IGameui();
        virtual void draw_game() = 0;
        virtual void handle_events() = 0;
    };

} // namespace rl::ui

#endif