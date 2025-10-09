#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <array>
#include <span>
#include <algorithm>
#include <ranges>
#include <concepts>
#include <string_view>

class Sokoban {
public:
    // Tile types
    static constexpr int WALL = 0;
    static constexpr int EMPTY = 1;
    static constexpr int PLAYER = 2;
    static constexpr int BOX = 3;
    static constexpr int TARGET = 4;
    static constexpr int BOX_ON_TARGET = 5;
    static constexpr int PLAYER_ON_TARGET = 6;
    
    // Actions
    static constexpr int UP = 0;
    static constexpr int DOWN = 1;
    static constexpr int LEFT = 2;
    static constexpr int RIGHT = 3;
    
    static constexpr int DEFAULT_SIZE = 10;
    static constexpr int MAX_ACTIONS = 4;

public:
    Sokoban();
    explicit Sokoban(int width, int height = DEFAULT_SIZE);
    
    void reset();
    void load_level(const std::string& level_str);
    // std::tuple<std::vector<int>, float, bool> step(int action);
    std::tuple<std::vector<int>, float, bool, bool> step(int action); // (obs, reward, done, box_pushed)
    std::vector<int> get_observation() const;
    std::vector<std::vector<int>> get_grid() const;
    bool is_solved() const;
    bool is_valid_action(int action) const;
    
    // Additional utility methods
    constexpr int get_width() const noexcept { return width_; }
    constexpr int get_height() const noexcept { return height_; }
    constexpr std::pair<int, int> get_player_position() const noexcept { return {player_x_, player_y_}; }

private:
    int width_, height_;
    std::vector<std::vector<int>> grid_;
    std::vector<std::vector<int>> initial_grid_;
    int player_x_, player_y_;
    int initial_player_x_, initial_player_y_;
    
    // Pre-allocated vectors to avoid dynamic allocation in step()
    mutable std::vector<int> observation_buffer_;
    
    // Helper methods
    static constexpr std::pair<int, int> get_direction_offset(int action) noexcept;
    constexpr bool is_valid_position(int x, int y) const noexcept;
    void find_player_position();
    void parse_level_string(std::string_view level_str);
    void create_default_level();
    void initialize_buffers();
    
    // Game logic helpers
    bool can_move_to(int x, int y) const noexcept;
    bool can_push_box_to(int x, int y) const noexcept;
    void move_player(int from_x, int from_y, int to_x, int to_y) noexcept;
    void move_box(int from_x, int from_y, int to_x, int to_y) noexcept;
};