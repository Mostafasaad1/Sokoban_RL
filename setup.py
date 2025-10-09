# setup.py
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

ext_modules = [
    Pybind11Extension(
        "sokoban_engine",  # ← This is your Python module name
        [
            "sokoban/sokoban.cpp",
            "sokoban/python_bindings.cpp"
        ],
        cxx_std=20,
        include_dirs=["sokoban/"],
    ),
]

setup(
    name="sokoban_rl",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "gymnasium",
        "pygame",
        "pybind11",
    ],
)
