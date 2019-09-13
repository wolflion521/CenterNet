from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="wolftest",
    ext_modules=[
        CppExtension("mypool", ["mypool.cpp"]),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)