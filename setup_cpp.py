from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

extra_compile_args = [
    "-O3",
    "-mavx2",
    "-mfma",
    "-fopenmp",
    "-march=native",
    "-std=c++17",
]

extra_link_args = [
    "-fopenmp",
]

setup(
    name="hybrid_cpp",
    ext_modules=[
        CppExtension(
            name="hybrid_cpp",
            sources=[
                "csrc/pybind_module.cpp",
                "csrc/engine.cpp",
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=["csrc"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
