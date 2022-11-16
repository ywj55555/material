from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv',
    ext_modules=[
        CUDAExtension('conv_cuda', [
            'conv_cuda.cpp',
            'conv_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })