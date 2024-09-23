#include "santorini_tournament_ui.hpp"
#include "../players_utils.hpp"
#include <raylib.h>
#include <thread>
#include <sstream>
#include <iostream>
namespace rl::ui
{
SantoriniTournamentUI::SantoriniTournamentUI(int width, int height)
	: board_width_{ width * 4 / 5 }, board_height_{ height * 4 / 5 }, padding_{ 2 },
	secondary_tap_width_{ width * 1 / 5 }, secondary_tap_height_{ height },
	footing_width_{ width * 4 / 5 }, footing_height_{ height * 1 / 5 },
	round_robin_ptr_(nullptr),
	state_ptr_(rl::games::SantoriniState::initialize_state()),
	players_{},
	selected_row_{ -1 },
	selected_col_{ -1 }

{
	round_robin_ptr_ = std::move(get_new_round_robin());
	std::function<void(rl::common::MatchInfo)> fn = std::bind(&SantoriniTournamentUI::on_state_changed, this, std::placeholders::_1);
	state_observer_ptr_ = round_robin_ptr_->matchinfo_changed_event.subscribe(fn);

	set_state_ptr(state_ptr_);

	cell_size_ = board_width_ / SantoriniState::ROWS;
	inner_cell_size_ = cell_size_ - 2 * padding_;
}

SantoriniTournamentUI::~SantoriniTournamentUI()
{
	state_observer_ptr_->unsubscribe();
}

void SantoriniTournamentUI::draw_game()
{
	draw_board();
	draw_players_name();
	draw_players_scores();
}

void SantoriniTournamentUI::handle_events()
{
	handle_board_events();
}

void SantoriniTournamentUI::init()
{

	round_robin_ptr_ = std::move(get_new_round_robin());
	std::function<void(rl::common::MatchInfo)> fn = std::bind(&SantoriniTournamentUI::on_state_changed, this, std::placeholders::_1);
	state_observer_ptr_ = round_robin_ptr_->matchinfo_changed_event.subscribe(fn);
	t_ptr_ = std::make_unique<std::thread>(&rl::common::RoundRobin::start, round_robin_ptr_.get());
	t_ptr_->detach();
}

void SantoriniTournamentUI::dispose()
{

}

std::unique_ptr<rl::common::RoundRobin> SantoriniTournamentUI::get_new_round_robin()
{
	players_.clear();
	constexpr int n_games_per_opponent = 2;
	auto players_duration = std::chrono::duration<int, std::milli>(1000);
	
	players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_strongest_220.pt"));
	players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_strongest_340.pt"));
	players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_strongest_380.pt"));
	players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_strongest_410.pt"));
	players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_strongest_470.pt"));
	players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_strongest_530.pt"));
	players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_590.pt"));
	players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_strongest_650.pt"));
	players_.push_back(get_long_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_long_80.pt"));
	players_.push_back(get_long_network_amcts2_player(state_ptr_.get(), 3, players_duration, "santorini_long_140.pt"));



	std::vector<rl::common::PlayerInfo> players_info{};
	for (auto& pi : players_)
	{
		players_info.push_back({ pi->player_ptr_.get(), pi->name_ });
	}
	auto round_robin = std::make_unique<rl::common::RoundRobin>(state_ptr_->clone(), players_info, n_games_per_opponent, false);
	return round_robin;
}

void SantoriniTournamentUI::set_state_ptr(std::unique_ptr<SantoriniState>& new_state_ptr)
{
	state_ptr_ = std::move(new_state_ptr->clone_state());
	obs_ = state_ptr_->get_observation();
	actions_legality_ = state_ptr_->actions_mask();
	phase_ = state_ptr_->get_current_phase();
	current_player_ = state_ptr_->player_turn();
	selected_row_ = -1;
	selected_col_ = -1;

	constexpr int selection_channel = 2;
	constexpr int channel_size = SantoriniState::ROWS * SantoriniState::COLS;
	constexpr int cell_start = channel_size * 2;
	constexpr int cell_end = cell_start + channel_size;
	for (int cell = cell_start; cell < cell_end; cell++)
	{
		if (obs_.at(cell) == 1.0f)
		{
			int index = cell - cell_start;
			selected_row_ = index / SantoriniState::COLS;
			selected_col_ = index % SantoriniState::COLS;
			break;
		}
	}
}

void SantoriniTournamentUI::on_state_changed(rl::common::MatchInfo match_info)
{
	auto new_state_ptr = match_info.state_ptr;
	auto casted_state_ptr = dynamic_cast<const rl::games::SantoriniState*>(new_state_ptr);
	if (casted_state_ptr == nullptr)
	{
		return;
	}

	player1_ = match_info.player_1_index;
	player2_ = match_info.player_2_index;
	set_state_ptr(casted_state_ptr->clone_state());
}

void SantoriniTournamentUI::draw_board()
{
	constexpr int ROWS = SantoriniState::ROWS;
	constexpr int COLS = SantoriniState::COLS;
	constexpr int CURRENT_PLAYER_CHANNEL = 0;
	constexpr int OPPONENT_PLAYER_CHANNEL = 1;
	constexpr int SELECTION_CHANNEL = 2;
	constexpr int GROUND_HEIGHT_CHANNEL = 3;
	constexpr int FLOOR1_CHANNEL = GROUND_HEIGHT_CHANNEL + 1;
	constexpr int FLOOR2_CHANNEL = FLOOR1_CHANNEL + 1;
	constexpr int FLOOR3_CHANNEL = FLOOR2_CHANNEL + 1;
	constexpr int DOME_CHANNEL = FLOOR3_CHANNEL + 1;
	constexpr int CHANNEL_SIZE = ROWS * COLS;
	int left, top, width, height, left_center, top_center;

	int current_player = current_player_;
	for (int row = 0; row < ROWS; row++)
	{
		for (int col = 0; col < COLS; col++)
		{
			left = col * cell_size_ + padding_;
			top = row * cell_size_ + padding_;
			left_center = left + inner_cell_size_ / 2;
			top_center = top + inner_cell_size_ / 2;

			int ground_ind = GROUND_HEIGHT_CHANNEL * CHANNEL_SIZE + row * COLS + col;
			int floor1_ind = FLOOR1_CHANNEL * CHANNEL_SIZE + row * COLS + col;
			int floor2_ind = FLOOR2_CHANNEL * CHANNEL_SIZE + row * COLS + col;
			int floor3_ind = FLOOR3_CHANNEL * CHANNEL_SIZE + row * COLS + col;
			int dome_ind = DOME_CHANNEL * CHANNEL_SIZE + row * COLS + col;

			if (obs_.at(ground_ind) == 1.0f)
			{
				draw_ground(left_center, top_center);
			}
			else if (obs_.at(floor1_ind) == 1.0f)
			{
				draw_floor1(left_center, top_center);
			}
			else if (obs_.at(floor2_ind) == 1.0f)
			{
				draw_floor2(left_center, top_center);
			}
			else if (obs_.at(floor3_ind) == 1.0f)
			{
				draw_floor3(left_center, top_center);
			}
			else if (obs_.at(dome_ind) == 1.0f)
			{
				draw_dome(left_center, top_center);
			}
			int player_0_channel = current_player == 0 ? 0 : 1;
			int player_1_channel = current_player == 0 ? 1 : 0;
			int player_0_ind = player_0_channel * CHANNEL_SIZE + row * COLS + col;
			int player_1_ind = player_1_channel * CHANNEL_SIZE + row * COLS + col;

			if (obs_.at(player_0_ind) == 1.0f)
			{
				draw_piece(left, top, 0, false);
			}
			else if (obs_.at(player_1_ind) == 1.0f)
			{
				draw_piece(left, top, 1, false);
			}

			Rectangle ol = { static_cast<float>(left), static_cast<float>(top), static_cast<float>(inner_cell_size_), static_cast<float>(inner_cell_size_) };
			if (selected_row_ == row && selected_col_ == col)
			{
				DrawRectangleLinesEx(ol, 2.0f, RED);
			}
		}
	}
	draw_legal_actions();
}

void SantoriniTournamentUI::draw_players_scores()
{

}

void SantoriniTournamentUI::draw_players_name()
{
	if (player1_ >= 0 && player2_ >= 0)
	{
		char text[200];
		std::stringstream ss{};
		ss << "[BLACK] " << players_.at(player1_)->name_;
		ss << " vs [WHITE] " << players_.at(player2_)->name_;
		auto a = ss.str();
		strcpy(text, a.c_str());
		SetWindowTitle(text);


		std::stringstream s1{};
		std::stringstream s2{};
		s1 << "[BLACK] " << players_.at(player1_)->name_;
		s2 << "[WHITE] " << players_.at(player2_)->name_;
		int top, left, width, height;
		top = board_height_;
		left = 0;
		width = footing_width_;
		height = footing_height_;
		int initial_font_size = 16;
		//DrawRectangle(0,top,width,height,RAYWHITE);
		DrawRectangle(0, top, width / 2, height, BLACK);
		DrawRectangle(width / 2, top, width / 2, height, WHITE);
		a = s1.str();
		strcpy(text, a.c_str());
		int font_size = initial_font_size + 2;
		int text_width = 1000000;
		while (text_width > width / 2)
		{
			font_size -= 2;
			text_width = MeasureText(text, font_size);
		}

		DrawText(text, (width / 2 - text_width) / 2, top + height / 2, font_size, GREEN);


		a = s2.str();
		strcpy(text, a.c_str());

		text_width = 1000000;
		font_size = initial_font_size + 2;
		while (text_width > width / 2)
		{
			font_size -= 2;
			text_width = MeasureText(text, font_size);
		}
		text_width = MeasureText(text, font_size);
		DrawText(text, (width / 2) + (width / 2 - text_width) / 2, top + height / 2, font_size, GREEN);

		if (current_player_ == 0)
		{
			// player 1 turn
			DrawRectangleLines(0, top, width / 2, height, RED);
		}
		else if (current_player_ == 1)
		{
			// player 2 turn
			DrawRectangleLines(width / 2, top, width / 2, height, RED);
		}

	}

}


void SantoriniTournamentUI::handle_board_events()
{
}

void SantoriniTournamentUI::draw_ground(int left_center, int top_center)
{
	int left = left_center - inner_cell_size_ / 2;
	int top = top_center - inner_cell_size_ / 2;
	DrawRectangle(left, top, inner_cell_size_, inner_cell_size_, DARKGREEN);
}

void SantoriniTournamentUI::draw_floor1(int left_center, int top_center)
{
	draw_ground(left_center, top_center);
	int size = inner_cell_size_ * 0.8;
	int left = left_center - size / 2;
	int top = top_center - size / 2;
	DrawRectangle(left, top, size, size, GRAY);
}

void SantoriniTournamentUI::draw_floor2(int left_center, int top_center)
{
	draw_floor1(left_center, top_center);
	int size = inner_cell_size_ * 0.8 * 0.8;
	int left = left_center - size / 2;
	int top = top_center - size / 2;
	DrawRectangle(left, top, size, size, DARKGRAY);
}

void SantoriniTournamentUI::draw_floor3(int left_center, int top_center)
{
	draw_floor2(left_center, top_center);
	int size = inner_cell_size_ * 0.8 * 0.8 * 0.8;
	int left = left_center - size / 2;
	int top = top_center - size / 2;
	DrawRectangle(left, top, size, size, RED);
}

void SantoriniTournamentUI::draw_dome(int left_center, int top_center)
{
	draw_floor3(left_center, top_center);
	int size = inner_cell_size_ * 0.8 * 0.8 * 0.8;
	// DrawRectangle(left, top, size, size, DARKBLUE);
	DrawCircle(left_center, top_center, size / 2, DARKBLUE);
}

void SantoriniTournamentUI::draw_piece(int left, int top, int player, bool is_fade)
{
	auto FADE_BLACK = BLACK;
	auto FADE_WHITE = WHITE;
	FADE_BLACK.a = 50;
	FADE_WHITE.a = 50;

	int left_center = left + inner_cell_size_ / 2;
	int top_center = top + inner_cell_size_ / 2;

	if (player == 0)

	{
		if (is_fade)
		{
			DrawCircle(left_center, top_center, inner_cell_size_ / 4, FADE_BLACK);
		}
		else
		{
			DrawCircle(left_center, top_center, inner_cell_size_ / 4, BLACK);
		}
	}
	else
	{
		if (is_fade)
		{
			DrawCircle(left_center, top_center, inner_cell_size_ / 4, FADE_WHITE);
		}
		else
		{
			DrawCircle(left_center, top_center, inner_cell_size_ / 4, WHITE);
		}
	}
}

void SantoriniTournamentUI::draw_legal_actions()
{
	int top, left;
	for (int action = 0; action < actions_legality_.size(); action++)
	{
		if (actions_legality_.at(action))
		{
			auto [row, col] = SantoriniState::decode_action(action);
			left = col * cell_size_ + padding_;
			top = row * cell_size_ + padding_;
			Rectangle ol = { left, top, inner_cell_size_, inner_cell_size_ };
			DrawRectangleLinesEx(ol, 2.0f, GREEN);
		}
	}
}

} // namespace rl::ui
