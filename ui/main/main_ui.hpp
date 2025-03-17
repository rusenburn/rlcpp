#ifndef RL_UI_MAIN_UI_HPP_
#define RL_UI_MAIN_UI_HPP_

#include "../IGameui.hpp"
#include "ui_windows.hpp"
#include <memory>
#include <common/observer.hpp>
#include <vector>
#include <raylib.h>
#include "../components/button.hpp"

namespace rl::ui
{
class MainUI : public IGameui
{
    using ButtonObserver = rl::common::Observer<int>;
public:
    MainUI(int width, int height);
    ~MainUI()override;
    void draw_game()override;
    void handle_events()override;

private:
    int width_, height_;
    std::unique_ptr<IGameui> game_ui_ptr_;
    std::vector<std::shared_ptr<ButtonObserver>> button_observers_;
    void handle_menu_events();
    void draw_menu();
    std::vector<std::unique_ptr<Button>> buttons_{};
    void on_button_1_clicked(int a);
    void on_button_2_clicked(int a);
    void on_button_3_clicked(int a);
    void on_button_4_clicked(int a);
};
} // namespace rl::ui



#endif