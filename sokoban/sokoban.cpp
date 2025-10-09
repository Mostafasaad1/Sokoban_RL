#include "sokoban.h"
#include <sstream>
#include <stdexcept>
#include <cassert>

// Define static constexpr members for pybind11
constexpr int Sokoban::WALL;
constexpr int Sokoban::EMPTY;
constexpr int Sokoban::PLAYER;
constexpr int Sokoban::BOX;
constexpr int Sokoban::TARGET;
constexpr int Sokoban::BOX_ON_TARGET;
constexpr int Sokoban::PLAYER_ON_TARGET;

constexpr int Sokoban::UP;
constexpr int Sokoban::DOWN;
constexpr int Sokoban::LEFT;
constexpr int Sokoban::RIGHT;

Sokoban::Sokoban() : Sokoban(DEFAULT_SIZE, DEFAULT_SIZE) {}

Sokoban::Sokoban(int width, int height) 
    : width_(width), height_(height), player_x_(0), player_y_(0),
      initial_player_x_(0), initial_player_y_(0) {
    
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }
    
    grid_.resize(height_, std::vector<int>(width_, EMPTY));
    
    // Only create default level if dimensions match default size
    if (width_ == DEFAULT_SIZE && height_ == DEFAULT_SIZE) {
        create_default_level();
        find_player_position();
    } else {
        // For custom dimensions, create a simple empty level with player at center
        if (width_ >= 3 && height_ >= 3) {
            // Add walls around the border
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    if (x == 0 || x == width_ - 1 || y == 0 || y == height_ - 1) {
                        grid_[y][x] = WALL;
                    }
                }
            }
            // Place player in center
            player_x_ = width_ / 2;
            player_y_ = height_ / 2;
            grid_[player_y_][player_x_] = PLAYER;
        } else {
            // Very small grid, just place player at (1,1) if possible
            player_x_ = std::min(1, width_ - 1);
            player_y_ = std::min(1, height_ - 1);
            grid_[player_y_][player_x_] = PLAYER;
        }
    }
    
    initial_grid_ = grid_;
    initial_player_x_ = player_x_;
    initial_player_y_ = player_y_;
    initialize_buffers();
}

void Sokoban::reset() {
    grid_ = initial_grid_;
    player_x_ = initial_player_x_;
    player_y_ = initial_player_y_;
}

void Sokoban::load_level(const std::string& level_str) {
    parse_level_string(level_str);
    initial_grid_ = grid_;
    find_player_position();
    initial_player_x_ = player_x_;
    initial_player_y_ = player_y_;
    initialize_buffers();
}

std::tuple<std::vector<int>, float, bool, bool> Sokoban::step(int action) {
    if (!is_valid_action(action)) {
        return {get_observation(), 0.0f, false, false};
    }
    
    auto [dx, dy] = get_direction_offset(action);
    const int new_x = player_x_ + dx;
    const int new_y = player_y_ + dy;
    
    assert(is_valid_position(new_x, new_y));
    
    const int target_tile = grid_[new_y][new_x];
    bool box_pushed = false;  // ← New flag
    
    // Handle wall collision
    if (target_tile == WALL) {
        return {get_observation(), 0.0f, false, false};
    }
    
    // Handle box pushing
    if (target_tile == BOX || target_tile == BOX_ON_TARGET) {
        const int box_new_x = new_x + dx;
        const int box_new_y = new_y + dy;
        
        if (!is_valid_position(box_new_x, box_new_y) || 
            !can_push_box_to(box_new_x, box_new_y)) {
            return {get_observation(), 0.0f, false, false};
        }
        
        // Execute box move
        move_box(new_x, new_y, box_new_x, box_new_y);
        box_pushed = true;  // ← Mark that a box was pushed
    }
    
    // Execute player move
    move_player(player_x_, player_y_, new_x, new_y);
    player_x_ = new_x;
    player_y_ = new_y;
    
    // Check win condition and calculate reward
    const bool solved = is_solved();
    const float reward = solved ? 1.0f : 0.0f;
    
    return {get_observation(), reward, solved, box_pushed};
}
std::vector<int> Sokoban::get_observation() const {
    // Use pre-allocated buffer to avoid allocation
    observation_buffer_.clear();
    observation_buffer_.reserve(width_ * height_);
    
    for (const auto& row : grid_) {
        observation_buffer_.insert(observation_buffer_.end(), row.begin(), row.end());
    }
    
    return observation_buffer_;
}

std::vector<std::vector<int>> Sokoban::get_grid() const {
    return grid_;
}

bool Sokoban::is_solved() const {
    bool has_boxes = false;
    bool has_unsolved_boxes = false;
    
    for (const auto& row : grid_) {
        for (int cell : row) {
            if (cell == BOX) {
                has_unsolved_boxes = true;
                has_boxes = true;
            } else if (cell == BOX_ON_TARGET) {
                has_boxes = true;
            }
        }
    }
    
    // Level is solved if there are boxes in the level but no unsolved boxes
    return has_boxes && !has_unsolved_boxes;
}

bool Sokoban::is_valid_action(int action) const {
    if (action < 0 || action >= MAX_ACTIONS) {
        return false;
    }
    
    auto [dx, dy] = get_direction_offset(action);
    const int new_x = player_x_ + dx;
    const int new_y = player_y_ + dy;
    
    return is_valid_position(new_x, new_y);
}

constexpr std::pair<int, int> Sokoban::get_direction_offset(int action) noexcept {
    switch (action) {
        case UP:    return {0, -1};
        case DOWN:  return {0, 1};
        case LEFT:  return {-1, 0};
        case RIGHT: return {1, 0};
        default:    return {0, 0};
    }
}

constexpr bool Sokoban::is_valid_position(int x, int y) const noexcept {
    return x >= 0 && x < width_ && y >= 0 && y < height_;
}

void Sokoban::find_player_position() {
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            if (grid_[y][x] == PLAYER || grid_[y][x] == PLAYER_ON_TARGET) {
                player_x_ = x;
                player_y_ = y;
                return;
            }
        }
    }
    throw std::runtime_error("Player not found in level");
}

void Sokoban::parse_level_string(std::string_view level_str) {
    std::vector<std::string_view> lines;
    
    // Split by newlines using ranges
    size_t start = 0;
    for (size_t end = level_str.find('\n'); end != std::string_view::npos; 
         start = end + 1, end = level_str.find('\n', start)) {
        lines.emplace_back(level_str.substr(start, end - start));
    }
    if (start < level_str.length()) {
        lines.emplace_back(level_str.substr(start));
    }
    
    if (lines.empty()) {
        throw std::invalid_argument("Empty level string");
    }
    
    height_ = static_cast<int>(lines.size());
    width_ = static_cast<int>(std::ranges::max_element(lines, {}, 
        [](const auto& line) { return line.length(); })->length());
    
    // Resize grid
    grid_.assign(height_, std::vector<int>(width_, EMPTY));
    
    for (int y = 0; y < height_; ++y) {
        const auto& row = lines[y];
        for (int x = 0; x < std::min(width_, static_cast<int>(row.length())); ++x) {
            switch (row[x]) {
                case '#': grid_[y][x] = WALL; break;
                case ' ': grid_[y][x] = EMPTY; break;
                case '@': grid_[y][x] = PLAYER; break;
                case '$': grid_[y][x] = BOX; break;
                case '.': grid_[y][x] = TARGET; break;
                case '*': grid_[y][x] = BOX_ON_TARGET; break;
                case '+': grid_[y][x] = PLAYER_ON_TARGET; break;
                default:  grid_[y][x] = EMPTY; break;
            }
        }
    }
}

void Sokoban::create_default_level() {
    // Create a simple 7x7 level with 2 boxes as requested
    constexpr std::string_view default_level = 
        "#######\n"
        "#  .  #\n"
        "# $ @ #\n"
        "#     #\n"
        "# $   #\n"
        "#  .  #\n"
        "#######";
    
    parse_level_string(default_level);
}

void Sokoban::initialize_buffers() {
    observation_buffer_.reserve(width_ * height_);
}

bool Sokoban::can_move_to(int x, int y) const noexcept {
    if (!is_valid_position(x, y)) return false;
    
    const int tile = grid_[y][x];
    return tile == EMPTY || tile == TARGET;
}

bool Sokoban::can_push_box_to(int x, int y) const noexcept {
    if (!is_valid_position(x, y)) return false;
    
    const int tile = grid_[y][x];
    return tile == EMPTY || tile == TARGET;
}

void Sokoban::move_player(int from_x, int from_y, int to_x, int to_y) noexcept {
    // Restore the tile where player was
    const int from_tile = grid_[from_y][from_x];
    if (from_tile == PLAYER_ON_TARGET) {
        grid_[from_y][from_x] = TARGET;
    } else {
        grid_[from_y][from_x] = EMPTY;
    }
    
    // Place player at new position
    const int to_tile = grid_[to_y][to_x];
    if (to_tile == TARGET) {
        grid_[to_y][to_x] = PLAYER_ON_TARGET;
    } else {
        grid_[to_y][to_x] = PLAYER;
    }
}

void Sokoban::move_box(int from_x, int from_y, int to_x, int to_y) noexcept {
    // Place box at new position
    const int to_tile = grid_[to_y][to_x];
    if (to_tile == TARGET) {
        grid_[to_y][to_x] = BOX_ON_TARGET;
    } else {
        grid_[to_y][to_x] = BOX;
    }
    
    // Restore the tile where box was (will be overwritten by player move)
    const int from_tile = grid_[from_y][from_x];
    if (from_tile == BOX_ON_TARGET) {
        grid_[from_y][from_x] = TARGET;
    } else {
        grid_[from_y][from_x] = EMPTY;
    }
}