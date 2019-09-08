from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="cpools",
    ext_modules=[
        CppExtension("top_pool", ["src/top_pool.cpp"]),
        CppExtension("bottom_pool", ["src/bottom_pool.cpp"]),
        CppExtension("left_pool", ["src/left_pool.cpp"]),
        CppExtension("right_pool", ["src/right_pool.cpp"])
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
# first call
# train.py :      nnet = NetworkFactory
# NetworkFactory inherited from kp
# kp.tl_cnvs is the output of make_tl_layer(256)
# tl_pool(256)
# go to pool(256,TopPool,LeftPool)
# go to TopPool
# go to TopPoolFunction
# go to top_pool in cpools/src/setup.py
# go to "src/top_pool.cpp"
