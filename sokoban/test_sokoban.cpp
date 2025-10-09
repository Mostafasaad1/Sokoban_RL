#include "sokoban.h"
#include <iostream>
#include <cassert>
#include <chrono>

// Simple test framework
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "...\n"; \
    test_##name(); \
    std::cout << "✓ " #name " passed\n"; \
} while(0)

TEST(basic_construction) {
    Sokoban game;
    assert(game.get_width() == 7);  // Default level is 7x7
    assert(game.get_height() == 7);
    assert(!game.is_solved());
}

TEST(custom_dimensions) {
    Sokoban game(5, 8);
    assert(game.get_width() == 5);
    assert(game.get_height() == 8);
}

TEST(level_loading) {
    Sokoban game;
    const std::string test_level = 
        "#####\n"
        "#@$.#\n"
        "#####";
    
    game.load_level(test_level);
    assert(game.get_width() == 5);
    assert(game.get_height() == 3);
    
    auto [px, py] = game.get_player_position();
    assert(px == 1 && py == 1);
}

TEST(basic_movement) {
    Sokoban game;
    const std::string simple_level = 
        "#####\n"
        "#@  #\n"
        "#####";
    
    game.load_level(simple_level);
    
    // Test valid movement
    auto [obs1, reward1, done1] = game.step(Sokoban::RIGHT);
    assert(reward1 == 0.0f);
    assert(!done1);
    
    auto [px, py] = game.get_player_position();
    assert(px == 2);
    
    // Test wall collision
    auto [obs2, reward2, done2] = game.step(Sokoban::UP);
    assert(reward2 == 0.0f);
    auto [px2, py2] = game.get_player_position();
    assert(px2 == 2);  // Should not move
}

TEST(box_pushing) {
    Sokoban game;
    const std::string level = 
        "######\n"
        "#@$  #\n"
        "######";
    
    game.load_level(level);
    
    // Push box to the right
    auto [obs, reward, done] = game.step(Sokoban::RIGHT);
    assert(reward == 0.0f);
    
    auto grid = game.get_grid();
    assert(grid[1][1] == Sokoban::EMPTY);     // Player was here
    assert(grid[1][2] == Sokoban::PLAYER);    // Player moved here
    assert(grid[1][3] == Sokoban::BOX);       // Box moved here
}

TEST(box_on_target) {
    Sokoban game;
    const std::string level = 
        "######\n"
        "#@$ .#\n"
        "######";
    
    game.load_level(level);
    
    // Push box onto target
    auto [obs1, reward1, done1] = game.step(Sokoban::RIGHT);
    assert(!done1);
    
    auto [obs2, reward2, done2] = game.step(Sokoban::RIGHT);
    assert(reward2 == 1.0f);  // Should be solved
    assert(done2);
    assert(game.is_solved());
}

TEST(invalid_actions) {
    Sokoban game;
    
    assert(!game.is_valid_action(-1));
    assert(!game.is_valid_action(4));
    assert(game.is_valid_action(0));
    assert(game.is_valid_action(3));
}

TEST(reset_functionality) {
    Sokoban game;
    const std::string level = 
        "######\n"
        "#@$  #\n"
        "######";
    
    game.load_level(level);
    auto [initial_px, initial_py] = game.get_player_position();
    
    // Move player
    game.step(Sokoban::RIGHT);
    auto [new_px, new_py] = game.get_player_position();
    assert(new_px != initial_px);
    
    // Reset and check
    game.reset();
    auto [reset_px, reset_py] = game.get_player_position();
    assert(reset_px == initial_px);
    assert(reset_py == initial_py);
}

TEST(observation_format) {
    Sokoban game;
    const std::string level = 
        "###\n"
        "#@#\n"
        "###";
    
    game.load_level(level);
    auto obs = game.get_observation();
    
    assert(obs.size() == 9);  // 3x3 grid
    assert(obs[4] == Sokoban::PLAYER);  // Center position
}

TEST(performance_benchmark) {
    Sokoban game;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run 10000 steps
    for (int i = 0; i < 10000; ++i) {
        game.step(i % 4);
        if (i % 100 == 0) {
            game.reset();
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "10000 steps took " << duration.count() << " microseconds\n";
    std::cout << "Average: " << (duration.count() / 10000.0) << " microseconds per step\n";
}

int main() {
    std::cout << "Running Sokoban Engine Tests\n";
    std::cout << "============================\n";
    
    try {
        RUN_TEST(basic_construction);
        RUN_TEST(custom_dimensions);
        RUN_TEST(level_loading);
        RUN_TEST(basic_movement);
        RUN_TEST(box_pushing);
        RUN_TEST(box_on_target);
        RUN_TEST(invalid_actions);
        RUN_TEST(reset_functionality);
        RUN_TEST(observation_format);
        RUN_TEST(performance_benchmark);
        
        std::cout << "\n✅ All tests passed!\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}