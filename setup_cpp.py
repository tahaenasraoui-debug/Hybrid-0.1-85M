import os
import platform
import shutil
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def _compiler_supports(flags: list[str], code: str = "int main() { return 0; }\n") -> bool:
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if not cxx:
        return False
    try:
        proc = subprocess.run(
            [cxx, "-x", "c++", *flags, "-c", "-", "-o", os.devnull],
            input=code,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


def _host_supports_avx2_fma() -> bool:
    arch = platform.machine().lower()
    if arch not in ("x86_64", "amd64"):
        return False
    cpuinfo_path = "/proc/cpuinfo"
    if os.path.exists(cpuinfo_path):
        try:
            with open(cpuinfo_path, "r", encoding="utf-8") as f:
                info = f.read()
            if "avx2" not in info or "fma" not in info:
                return False
        except OSError:
            pass
    return _compiler_supports(["-mavx2", "-mfma", "-march=native"])


extra_compile_args = ["-O3", "-std=c++17"]
extra_link_args: list[str] = []

if os.environ.get("HYBRID_PORTABLE", "0") != "1" and _host_supports_avx2_fma():
    extra_compile_args.extend(["-mavx2", "-mfma", "-march=native"])

if _compiler_supports(["-fopenmp"], code="#include <omp.h>\nint main() { return omp_get_max_threads(); }\n"):
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")

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
