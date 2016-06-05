import os
from Cython.Distutils import build_ext
from setuptools import setup
import numpy as np
from distutils.extension import Extension

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "cython_bbox",
        ["bbox.pyx"],
        extra_compile_args= ["-Wno-cpp", "-Wno-unused-function"],
        include_dirs = [numpy_include])
]

setup(
    name='fast_rcnn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
