# Minimal Makefile for Sokoban Engine

CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra
PYTHON_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_SUFFIX = $(shell python3-config --extension-suffix)

# Default target
all: test_sokoban sokoban_engine$(PYTHON_SUFFIX)

# C++ test executable
test_sokoban: test_sokoban.cpp sokoban.cpp sokoban.h
	$(CXX) $(CXXFLAGS) test_sokoban.cpp sokoban.cpp -o test_sokoban

# Python module
sokoban_engine$(PYTHON_SUFFIX): python_bindings.cpp sokoban.cpp sokoban.h
	$(CXX) $(CXXFLAGS) -shared -fPIC $(PYTHON_INCLUDES) \
		python_bindings.cpp sokoban.cpp -o sokoban_engine$(PYTHON_SUFFIX)

# Object file for library usage
sokoban.o: sokoban.cpp sokoban.h
	$(CXX) $(CXXFLAGS) -c sokoban.cpp -o sokoban.o

# Run tests
test: test_sokoban
	./test_sokoban

# Run Python example (requires Python module)
example: sokoban_engine$(PYTHON_SUFFIX)
	python3 example_usage.py

# Clean build artifacts
clean:
	rm -f test_sokoban sokoban.o sokoban_engine$(PYTHON_SUFFIX)

# Install Python dependencies
install-deps:
	pip install pybind11 numpy matplotlib pillow

# Demo visualization
demo-viz: sokoban_engine$(PYTHON_SUFFIX)
	python3 demo_visualization.py

# Interactive visualization
interactive: sokoban_engine$(PYTHON_SUFFIX)
	python3 visualize_sokoban.py

# RL training demo
rl-demo: sokoban_engine$(PYTHON_SUFFIX)
	python3 rl_visualization_demo.py

.PHONY: all test example clean install-deps demo-viz interactive rl-demo