# -*- coding: utf-8 -*-
from setuptools import setup
from typing import Any, Dict
from setuptools_cpp import CMakeExtension, ExtensionBuilder, Pybind11Extension


setup_kwargs = {}
ext_modules = [
    Pybind11Extension(
        "hetgpy.matern", ["hetgpy/src/matern.cpp"], 
        include_dirs=["eigen/"]
    ),
    Pybind11Extension(
        "hetgpy.qEI", ["hetgpy/src/qEI.cpp"], 
        include_dirs=["eigen/"]
    ),
    Pybind11Extension(
        "hetgpy.EMSE", ["hetgpy/src/EMSE.cpp"], 
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
build(setup_kwargs)

setup(**setup_kwargs)