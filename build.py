from glob import glob
from typing import Any, Dict
from setuptools import Extension

ext_modules = [
    # A basic pybind11 extension in <project_root>/src/ext1:
    Extension(
        name="hetgpy.src.gauss", 
        sources=["hetgpy/src/gauss.cpp"], 
        include_dirs=["eigen"]
    )
]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules, 
            "zip_safe": False,
        }
    )
