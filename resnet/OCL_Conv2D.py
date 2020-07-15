import torch
import torch.nn as nn
import os
import numpy as np

import pyopencl as cl

class OCL_Conv2D(nn.Conv2d):
    
    """
    Convolution implemented with OpenCL


    in_channels - Number of channels in the input image
    out_channels - Number of channels produced by the convolution

    use_ocl - Boolean, if set to True, will use OpenCL Convolution instead of PyTorch implementaition


    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_ocl=False):
        self.use_ocl = use_ocl
        super(OCL_Conv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.context = self.getOclContext()
        self.programm = self.getOCLprogramm(self.context)

    def getOCLprogramm(self, oclContext):
        
        PATH_TO_KERNEL = 'opencl/conv2d.cl'

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
            Given two arrays representing the dimension-size 
            for an input and kernel, calculates the dimension-size
            when performing a convolution.
        """
        output_height = (input_dim[0] - kernel_dim[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_width = (input_dim[1] - kernel_dim[1] + 2 * self.padding[1]) // self.stride[1] + 1
        return np.array([output_height, output_width], dtype=np.int32)

    def getOutputDimensions(self, input):
        assert(len(input.shape) == 2)
        output_width = (input.shape[1] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        output_height = (input.shape[0] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        return np.array([output_height, output_width], dtype=np.int32)

    def performOCLconvolution(self, input, weight):
        """
            Takes input and weight as torch.tensor type 
            and performs the correct convolution on them
            to produce a torch.tensor type result
            Input tensor has to be (channels, height, width)
            Weights are (out_channels, in_channels , height, width)
            WARNING: do not use 4D tensor with batchsize
        """
        assert(len(input.shape) == 3)
        assert(len(weight.shape) == 4)

        print("Input Channels:  ", self.in_channels)
        print("Output Channels: ", self.out_channels)
        print("Weight shape:    ", weight.shape)
        print("Input shape:     ", input.shape)

        output_dim = self.getOutputDimensions(input[0])
        
        result_tensor = []
        channel_count = 0
        for channel_weights in weight:
            channel_count += 1
            print("Computing Channel:  ", channel_count)


            channel_result_tensor = []

            for kernel_plane in channel_weights:

                channel_output_plane = np.zeros(output_dim, dtype=np.float32)

                for input_plane in input:
                    temp_res = self.OCLconvolution2D(
                        input_plane.detach().numpy(),
                        kernel_plane.detach().numpy()
                    )
                    channel_output_plane = channel_output_plane.__add__(temp_res)
                
                channel_result_tensor.append(channel_output_plane)

            # append result of the output channel
            result_tensor.append(torch.tensor(channel_result_tensor[0]))

        return torch.stack(result_tensor)

    def OCLconvolution2D(self, input_2d, kernel_2d):
        """
            Takes 2 dimensional input_2d and 2 dimensional
            filter as numpy array to perform the convolution with
            Returns 2 dimensional numpy-array result
        """
        assert(len(input_2d.shape) == 2)
        assert(len(kernel_2d.shape) == 2)

        # context and programm
        #ctx = self.getOclContext()
        #prg = self.getOCLprogramm(ctx)
        ctx = self.context
        prg = self.programm
        
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # conversion of data types and creation of buffers

        # 1. Input data, with dimensions
        np_x = input_2d
        if self.padding != 0:
            np.zeros((np_x.shape[0] + self.padding[0], np_x.shape[1] + self.padding[1]), dtype=np.int32)

        np_dim_x = np.array(np_x.shape, dtype=np.int32)

        buffer_x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_x)
        buffer_dim_x = cl.Buffer(ctx ,mf.READ_ONLY |mf.COPY_HOST_PTR, hostbuf=np_dim_x)

        # 2. kernel, with dimensions
        np_kernel = kernel_2d
        np_dim_kernel = np.array(np_kernel.shape, dtype=np.int32)

        buffer_kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_kernel)
        buffer_dim_kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dim_kernel)

        # 3. Result buffer
        np_dim_output = self.getNumpyOutputDimensions(np_dim_x, np_dim_kernel)
        np_output = np.zeros(np_dim_output, dtype=np.float32)
        
        buffer_output = cl.Buffer(ctx, mf.READ_WRITE, np_output.nbytes)

        # 4. Stride buffer
        buffer_stride = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array((self.stride, self.stride), dtype=np.int32))

        # OpenCL kernel, executed for single result
        convolutionFunct = prg.conv2d2
        convolutionFunct.set_args(
            buffer_kernel, 
            buffer_dim_kernel, 
            buffer_x,
            buffer_dim_x,
            buffer_output,
            buffer_stride)

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
            return super(OCL_Conv2D, self).forward(x)
        
