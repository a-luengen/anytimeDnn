import torch
import torch.nn as nn
import os
import numpy as np

import pyopencl as cl

class OCL_Convolution(nn.Conv2d):
    
    """
    Convolution implemented with OpenCL

    in_channels - Number of channels in the input image
    out_channels - Number of channels produced by the convolution

    use_ocl - Boolean, if set to True, will use OpenCL Convolution instead of PyTorch implementaition

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', use_ocl=False):
        self.use_ocl = use_ocl
        super(OCL_Convolution, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if use_ocl: 
            self.context = self.getOclContext()
            self.programm = self.getOCLprogramm(self.context)

    def getOCLprogramm(self, oclContext):
        
        PATH_TO_KERNEL = 'opencl/convTensor.cl'

        # prevent using cached source code, as this may cause 
        # "compiler caching failed exception"
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['PYOPENCL_NO_CACHE'] = '1' 
        os.environ['PYOPENCL_CTX'] = '0'

        # read kernel file
        f = open(PATH_TO_KERNEL, 'r', encoding='utf-8')
        kernels = ' '.join(f.readlines())
        f.close()

        return cl.Program(oclContext, kernels).build()

    def getOclContext(self):
        platforms = cl.get_platforms()
        return cl.Context(
            dev_type=cl.device_type.ALL,
            properties=[(cl.context_properties.PLATFORM, platforms[0])]
        )

    def getNumpyOutputDimensions(self, input_dim, kernel_dim):
        """
            Given two arrays representing the dimension-size (shape)
            for an input and kernel, calculates the dimension-size (shape)
            when performing a convolution.
            Input: (channel, height, width)
            kernel: (in_channel, height, width)
        """
        assert(len(input_dim) == 3)
        assert(len(kernel_dim) == 4)
        output_height = (input_dim[1] - kernel_dim[2] + 2 * self.padding[0]) // self.stride[0] + 1
        output_width = (input_dim[2] - kernel_dim[3] + 2 * self.padding[1]) // self.stride[1] + 1
        return np.array([self.out_channels, output_height, output_width], dtype=np.int32)

    def performOCLconvolution(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
            Takes input and weight as torch.tensor type 
            and performs the correct convolution on them
            to produce a torch.tensor type result
            WARNING: do not use 4D tensor with batchsize

            Parameters
            ----------
            input: torch.Tensor
                Tensor from the Input to perform the convolution on.
                (in_channels, height, width)
            weight: torch.Tensor
                Internal weights of this Convolution-Layer.
                (out_channels, in_channels, height, width)
        """

        assert (len(input.shape) == 3), "Input shapes size was {}".format(len(input.shape))
        assert (len(weight.shape) == 4)
        
        bias = np.zeros_like(self.bias) if self.bias is None else self.bias.detach().numpy()
        out_result = self.OCLconvolution(input.detach().numpy(), weight.detach().numpy(), bias)
        return torch.tensor(out_result)

    def OCLconvolution(self, input_3d: np.ndarray, kernel_4d: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """
            Takes 3 dimensional input_3d and 3 dimensional
            weight as numpy array to perform the convolution with
            Returns 3 dimensional numpy-array as result

            Parameters
            ----------
            input_3d: np.ndarray
                3 dimensional numpy array representing values from a tensor containing
                values of a single batch from the input.
                (in_channels, height, width)

        """

        assert(len(input_3d.shape) == 3)
        assert(len(kernel_4d.shape) == 4)
        if self.bias is not None:
            assert(len(bias.shape) == 1)

        # context and programm
        ctx = self.context
        prg = self.programm
        
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # conversion of data types and creation of buffers

        # 1. Input data, with dimensions
        np_x = input_3d
        if self.padding[0] > 0 or self.padding[1] > 0:
            pad_val = 0.0 if self.padding_mode == 'zeros' else 1.0
            np_x = np.pad(np_x, ((0,0), self.padding, self.padding), mode='constant', constant_values=0.0)


        np_dim_x = np.array(np_x.shape, dtype=np.int32)

        buffer_x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_x)
        buffer_dim_x = cl.Buffer(ctx ,mf.READ_ONLY |mf.COPY_HOST_PTR, hostbuf=np_dim_x)

        # 2. kernel with dimensions
        np_kernel = kernel_4d
        np_dim_kernel = np.array(np_kernel.shape, dtype=np.int32)

        buffer_kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_kernel)
        buffer_dim_kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dim_kernel)

        # 3. Output buffer
        np_dim_output = self.getNumpyOutputDimensions(input_3d.shape, kernel_4d.shape)
        np_output = np.zeros(np_dim_output, dtype=np.float32)
        
        buffer_output = cl.Buffer(ctx, mf.READ_WRITE, np_output.nbytes)
        buffer_dim_output = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dim_output)
        # 4. Stride buffer
        buffer_stride = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array((self.stride, self.stride), dtype=np.int32))

        # 5. bias buffer
        buffer_bias = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(bias))

        # OpenCL kernel, executed for single result
        convolutionFunct = prg.convolution
        convolutionFunct.set_args(
            buffer_kernel, 
            buffer_dim_kernel, 
            buffer_x,
            buffer_dim_x,
            buffer_output,
            buffer_dim_output,
            buffer_stride,
            buffer_bias)
        cl.enqueue_nd_range_kernel(queue, convolutionFunct, np_output.shape, None)
        cl.enqueue_copy(queue, np_output, buffer_output)
        return np_output


    def ocl_conv2d_forward(self, input):
        result = []
        for batch in input:
            tempRes = self.performOCLconvolution(batch, self.weight)
            result.append(tempRes)
        return torch.stack(result)

    def forward(self, x):
        if self.use_ocl == True:
            return self.ocl_conv2d_forward(x)
        else:
            return super(OCL_Convolution, self).forward(x)
        
