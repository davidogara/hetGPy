from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "distance_cpp",
        ['src/distcpp.cpp'],
    )
    
]

setup(cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)