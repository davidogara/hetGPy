# build.py

from typing import Any, Dict

from setuptools_cpp import CMakeExtension, ExtensionBuilder, Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "hetgpy.gauss", ["hetgpy/src/gauss.cpp"], 
        include_dirs=["eigen/"]
    ),
    Pybind11Extension(
        "hetgpy.matern", ["hetgpy/src/matern.cpp"], 
        include_dirs=["eigen/"]
    )
]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": dict(build_ext=ExtensionBuilder),
            "zip_safe": False,
        }
    )