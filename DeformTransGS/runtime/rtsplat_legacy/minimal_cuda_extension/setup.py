from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(
    name='add_one_cuda',
    ext_modules=[CUDAExtension('add_one_cuda', ['add_one_kernel.cu'])],
    cmdclass={'build_ext': BuildExtension},
)
