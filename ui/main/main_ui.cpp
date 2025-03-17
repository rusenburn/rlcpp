#include "main_ui.hpp"
#include "../santorini/santorini_ui.hpp"
#include "../walls/walls_ui.hpp"
#include "../damma/damma_ui.hpp"
#include "../othello/othello_ui.hpp"
#include <functional>
namespace rl::ui
{
MainUI::MainUI(int width, int height)
    :width_(width), height_(height), game_ui_ptr_{ nullptr }
{
    Rectangle rect;
    Color color{ GRAY };
    float padding_top = 20;
    float button_width = 100;
    float button_height = 20;
    float top = 0;
    float left;
    left = (width_ - button_width) / 2;
    
    top += padding_top;
    rect = { left, top, button_width, button_height };
    buttons_.push_back(std::move(std::make_unique<Button>("Santorini", rect, color)));
    std::function<void(int)> fn = std::bind(&MainUI::on_button_1_clicked, this, std::placeholders::_1);
    button_observers_.push_back(buttons_.back()->subscribe(fn));
    
    top += padding_top + button_height;
    rect = { left, top, button_width, button_height };
    buttons_.push_back(std::move(std::make_unique<Button>("Damma", rect, color)));
    fn = std::bind(&MainUI::on_button_2_clicked, this, std::placeholders::_1);
    button_observers_.push_back(buttons_.back()->subscribe(fn));
    
    top += padding_top + button_height;
    rect = { left, top, button_width, button_height };
    buttons_.push_back(std::move(std::make_unique<Button>("Walls", rect, color)));
    fn = std::bind(&MainUI::on_button_3_clicked, this, std::placeholders::_1);
    button_observers_.push_back(buttons_.back()->subscribe(fn));

    top += padding_top + button_height;
    rect = { left, top, button_width, button_height };
    buttons_.push_back(std::move(std::make_unique<Button>("Othello", rect, color)));
    fn = std::bind(&MainUI::on_button_4_clicked, this, std::placeholders::_1);
    button_observers_.push_back(buttons_.back()->subscribe(fn));
    // subscribe to button
}

MainUI::~MainUI() = default;

void MainUI::draw_game() {
    if (game_ui_ptr_ == nullptr)
    {
        draw_menu();
    }
    else {
        game_ui_ptr_->draw_game();
    }
}

void MainUI::handle_events()
{
    if (game_ui_ptr_ == nullptr)
    {
        handle_menu_events();
    }
    else {
        game_ui_ptr_->handle_events();
    }
}

void MainUI::handle_menu_events()
{
    for (auto& button : buttons_)
    {
        button->handle_events();
    }
}

void MainUI::draw_menu()
{
    for (auto& button : buttons_)
    {
        button->draw();
    }
}


void MainUI::on_button_1_clicked(int a) {
    this->game_ui_ptr_ = std::move(std::make_unique<SantoriniUI>(width_, height_));
}

void MainUI::on_button_2_clicked(int a) {
    this->game_ui_ptr_ = std::move(std::make_unique<DammaUI>(width_, height_));
}

void MainUI::on_button_3_clicked(int a) {
    this->game_ui_ptr_ = std::move(std::make_unique<WallsUi>(width_, height_));
}

void MainUI::on_button_4_clicked(int a) {
    this->game_ui_ptr_ = std::move(std::make_unique<OthelloUI>(width_, height_));
}

} // namespace rl::ui
