from torch.utils.cpp_extension import load

conv_cuda = load(
    'conv_cuda', ['conv_cuda.cpp', 'conv_cuda_kernel.cu'], verbose=True)
help(conv_cuda)
