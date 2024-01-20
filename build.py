from pybind11.setup_helpers import Pybind11Extension, build_ext
def build(setup_kwargs):
    ext_modules = [
        Pybind11Extension(
            "gauss",
            ["hetgpy/gauss.cpp"],
            include_dirs=[".", "eigen/"],
            extra_compile_args=['-O3'],
            language='c++',
            cxx_std=17
        ),
        Pybind11Extension(
            "matern",
            ["hetgpy/matern.cpp"],
            include_dirs=[".", "eigen/"],
            extra_compile_args=['-O3'],
            language='c++',
            cxx_std=17
        ),
    ]
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmd_class": {"build_ext": build_ext},
        "zip_safe": False,
    })