"""
    Source: https://github.com/tmehari/ssm_ecg 
"""

from setuptools import setup, find_packages
import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'cauchy_mult', [
            'cauchy.cpp',
            'cauchy_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-march=native', '-funroll-loops'],
                            # 'nvcc': ['-O2', '-lineinfo']
                            'nvcc': ['-O2', '-lineinfo', '--use_fast_math']
                            }
    )
    ext_modules.append(extension)

#setup(
#    name='cauchy_mult',
#    ext_modules=ext_modules,
#    # cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)})
#    cmdclass={'build_ext': BuildExtension})
setup(name='cauchy-mult',
      version='0.0',
      packages=find_packages(), #fix
      description='Sequential model-based optimization toolbox.', 
      cmdclass={'build_ext':BuildExtension})
