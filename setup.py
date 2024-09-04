# -*- coding: utf-8 -*-
from setuptools import setup
from typing import Any, Dict
from setuptools_cpp import CMakeExtension, ExtensionBuilder, Pybind11Extension

import requests
import os
import shutil


output = 'eigen'
os.makedirs(output,exist_ok=True)

url = r'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz'
zipname = url.split('/')[-1]
zippath = os.path.join(output,zipname)

# if already downloaded, skip
if not os.path.exists(zippath):
    
    # download
    r = requests.get(url)
    with open(zippath, 'wb') as f:
        f.write(r.content)
else:
    print(f'{zippath} exists, skipping download')

shutil.unpack_archive(zippath,output)
for item in os.listdir('eigen/eigen-3.4.0'):
    src = 'eigen/eigen-3.4.0/' + item
    dst = 'eigen'
    if not os.path.exists(src):
        shutil.move(src=src,dst=dst)
shutil.rmtree('eigen/eigen-3.4.0')


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