from torch.utils.cpp_extension import load
import torch as tt
from torch import nn
import torch.nn.functional as F

conv_cuda = load(
    'conv_cuda', ['conv_cuda.cpp', 'conv_cuda_kernel.cu'], verbose=True)


# help(conv_cuda)``1        1`

def check_forward():
    batch_size = 32
    in_channels = 64
    out_channels = 64
    in_size = 256
    kernel_size = 3
    padding = 1
    stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    weight = tt.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
    bias = tt.randn(out_channels).cuda()
    output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    conv_cuda.forward(input,
                      weight,
                      bias,
                      output,
                      kernel_size, kernel_size,
                      stride, stride,
                      padding, padding)
    input.clone()
    weight.clone()
    output.clone()

    # print(output)
    # print(input)
    # print(F.conv2d(input, weight, bias, padding=padding))
    # out_ref = F.conv2d(input, weight, bias, padding=padding)

    import time

    time_b = time.time()
    conv_cuda.forward(input,
                      weight,
                      bias,
                      output,
                      kernel_size, kernel_size,
                      stride, stride,
                      padding, padding)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    time_b = time.time()
    out_ref = F.conv2d(input, weight, bias, padding=padding)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    print("max error: {:.3e}".format(float((out_ref - output).abs().max())))

    time_b = time.time()
    conv_cuda.forward(input,
                      weight,
                      bias,
                      output,
                      kernel_size, kernel_size,
                      stride, stride,
                      padding, padding)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    time_b = time.time()
    out_ref = F.conv2d(input, weight, bias, padding=padding)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    print("max error: {:.3e}".format(float((out_ref - output).abs().max())))


def check_grad_in():
    batch_size = 10
    in_channels = 32
    out_channels = 32
    in_size = 256
    kernel_size = 3
    padding = 1
    stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    input.requires_grad = True
    weight = tt.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
    bias = tt.randn(out_channels).cuda()
    output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    outref = F.conv2d(input, weight, bias, padding=padding)
    outref.backward(output)

    grad_clone = input.grad.clone()

    conv_cuda.backward_input(output,
                             weight,
                             input,
                             kernel_size, kernel_size,
                             stride, stride,
                             padding, padding)

    print(((grad_clone - input)).abs().max())


def check_grad_weight():
    batch_size = 32
    in_channels = 3
    out_channels = 6
    in_size = 256
    kernel_size = 3
    padding = 1
    stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    weight = tt.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
    weight.requires_grad = True
    bias = tt.randn(out_channels).cuda()
    output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    outref = F.conv2d(input, weight, bias, padding=padding)
    outref.backward(output)

    grad_clone = weight.grad.clone()

    weight2 = weight.clone()
    conv_cuda.backward_weight(output,
                              input,
                              weight2,
                              kernel_size, kernel_size,
                              stride, stride,
                              padding, padding)
    eps = 1e-6
    print(((grad_clone - weight2) / (grad_clone.abs() + eps)).abs().max())

    import time
    time_b = time.time()
    conv_cuda.backward_weight(output,
                              input,
                              weight2,
                              kernel_size, kernel_size,
                              stride, stride,
                              padding, padding)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    outref = F.conv2d(input, weight, bias, padding=padding)
    time_b = time.time()
    outref.backward(output)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    time_b = time.time()
    conv_cuda.backward_weight(output,
                              input,
                              weight2,
                              kernel_size, kernel_size,
                              stride, stride,
                              padding, padding)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    outref = F.conv2d(input, weight, bias, padding=padding)
    time_b = time.time()
    outref.backward(output)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    time_b = time.time()
    conv_cuda.backward_weight(output,
                              input,
                              weight2,
                              kernel_size, kernel_size,
                              stride, stride,
                              padding, padding)
    time_e = time.time()
    print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    outref = F.conv2d(input, weight, bias, padding=padding)
    time_b = time.time()
    outref.backward(output)
    time_e = time.time()
    print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))


def check_naive_clone():
    batch_size = 1
    in_channels = 1
    out_channels = 1
    in_size = 3
    kernel_size = 1
    padding = 0
    stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    weight = tt.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
    bias = tt.randn(out_channels).cuda()
    output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    result = conv_cuda.forward(input,
                               weight,
                               bias,
                               output,
                               kernel_size, kernel_size,
                               stride, stride,
                               padding, padding)
    print(result)
    input.clone()
    weight.clone()
    bias.clone()
    output.clone()

    F.conv2d(input, weight, bias, padding=padding)

    # input.clone()


if __name__ == '__main__':
    check_forward()
    # check_grad_in()
    # check_grad_weight()
    # check_naive_clone()
