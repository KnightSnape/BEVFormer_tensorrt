import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import time


def get_logger(level=trt.Logger.INTERNAL_ERROR):
    TRT_LOGGER = trt.Logger(level)
    return TRT_LOGGER


def create_engine_context(trt_model, trt_logger):
    with open(trt_model, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


class HostDeviceMem(object):
    def __init__(self, name, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.name = name
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return (
            "Name:\n"
            + str(self.name)
            + "\nHost:\n"
            + str(self.host)
            + "\nDevice:\n"
            + str(self.device)
            + "\n"
        )

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, context, input_shapes, output_shapes):
    inputs = []
    outputs = []
    bindings = []
    
    # TensorRT 10+ uses different API
    # Use num_io_tensors instead of num_bindings
    try:
        num_tensors = engine.num_io_tensors
        use_new_api = True
    except AttributeError:
        # Fallback to old API for TensorRT < 10
        num_tensors = len(list(engine))
        use_new_api = False
    
    for i in range(num_tensors):
        if use_new_api:
            # TensorRT 10+ API
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            is_input = (mode == trt.TensorIOMode.INPUT)
            dtype_trt = engine.get_tensor_dtype(name)
            
            if is_input:
                dims = input_shapes[name]
                context.set_input_shape(name, dims)
            else:
                dims = output_shapes[name]
        else:
            # Old API for TensorRT < 10
            binding = list(engine)[i]
            name = binding
            is_input = engine.binding_is_input(binding)
            dtype_trt = engine.get_binding_dtype(binding)
            
            if is_input:
                dims = input_shapes[name]
                context.set_binding_shape(i, dims)
            else:
                dims = output_shapes[name]
        
        size = trt.volume(dims)
        dtype = trt.nptype(dtype_trt)
        assert dtype == np.float32, f"Engine's inputs/outputs only support FP32, but got {dtype} for {name}."
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        
        if is_input:
            inputs.append(HostDeviceMem(name, host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(name, host_mem, device_mem))
    
    return inputs, outputs, bindings


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to device
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    stream.synchronize()
    
    t1 = time.time()
    
    # TensorRT 10+ uses execute_async_v3 with tensor addresses
    # For TensorRT 10, we need to set tensor addresses using context.set_tensor_address
    # Check if new API is available
    if hasattr(context, 'execute_async_v3'):
        # TensorRT 10+ API: set tensor addresses before execution
        for inp in inputs:
            context.set_tensor_address(inp.name, int(inp.device))
        for out in outputs:
            context.set_tensor_address(out.name, int(out.device))
        
        # Execute inference
        context.execute_async_v3(stream_handle=stream.handle)
    else:
        # Old API (TensorRT < 10)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    stream.synchronize()
    t2 = time.time()
    
    # Transfer output data to host
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    return outputs, t2 - t1
