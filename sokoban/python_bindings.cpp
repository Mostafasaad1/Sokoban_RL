#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sokoban.h"

namespace py = pybind11;

PYBIND11_MODULE(sokoban_engine, m) {
    m.doc() = "High-performance Sokoban game engine for reinforcement learning";
    
    // Define the constants in the module
    m.attr("WALL") = Sokoban::WALL;
    m.attr("EMPTY") = Sokoban::EMPTY;
    m.attr("PLAYER") = Sokoban::PLAYER;
    m.attr("BOX") = Sokoban::BOX;
    m.attr("TARGET") = Sokoban::TARGET;
    m.attr("BOX_ON_TARGET") = Sokoban::BOX_ON_TARGET;
    m.attr("PLAYER_ON_TARGET") = Sokoban::PLAYER_ON_TARGET;

    m.attr("UP") = Sokoban::UP;
    m.attr("DOWN") = Sokoban::DOWN;
    m.attr("LEFT") = Sokoban::LEFT;
    m.attr("RIGHT") = Sokoban::RIGHT;

    py::class_<Sokoban>(m, "Sokoban")
        .def(py::init<>())
        .def(py::init<int, int>(), "Construct with custom dimensions", 
             py::arg("width"), py::arg("height") = Sokoban::DEFAULT_SIZE)
        
        // Core game methods
        .def("reset", &Sokoban::reset, "Reset to initial state")
        .def("load_level", &Sokoban::load_level, "Load level from string", py::arg("level_str"))
        .def("step", &Sokoban::step, "Execute action and return (observation, reward, done)", 
             py::arg("action"))
        .def("get_observation", &Sokoban::get_observation, "Get flattened grid observation")
        .def("get_grid", &Sokoban::get_grid, "Get 2D grid for rendering")
        .def("is_solved", &Sokoban::is_solved, "Check if level is solved")
        .def("is_valid_action", &Sokoban::is_valid_action, "Check if action is valid", 
             py::arg("action"))
        
        // Utility methods
        .def("get_width", &Sokoban::get_width, "Get grid width")
        .def("get_height", &Sokoban::get_height, "Get grid height")
        .def("get_player_position", &Sokoban::get_player_position, "Get player (x, y) position")
        
        // Constants
        .def_readonly_static("WALL", &Sokoban::WALL)
        .def_readonly_static("EMPTY", &Sokoban::EMPTY)
        .def_readonly_static("PLAYER", &Sokoban::PLAYER)
        .def_readonly_static("BOX", &Sokoban::BOX)
        .def_readonly_static("TARGET", &Sokoban::TARGET)
        .def_readonly_static("BOX_ON_TARGET", &Sokoban::BOX_ON_TARGET)
        .def_readonly_static("PLAYER_ON_TARGET", &Sokoban::PLAYER_ON_TARGET)
        
        .def_readonly_static("UP", &Sokoban::UP)
        .def_readonly_static("DOWN", &Sokoban::DOWN)
        .def_readonly_static("LEFT", &Sokoban::LEFT)
        .def_readonly_static("RIGHT", &Sokoban::RIGHT);
}