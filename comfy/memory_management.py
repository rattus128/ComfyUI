import torch
from comfy.quant_ops import QuantizedTensor

from comfy.cli_args import args
import comfy.model_management

import ctypes
import aimdo.control

import logging

def get_pin(module):
    return getattr(module, "_pin", None)

PIN_TOTAL=0

def pin_memory(module):
    global PIN_TOTAL
    if module.pin_failed or args.disable_pinned_memory or get_pin(module) is not None:
        return
    #FIXME: This is a RAM cache trigger event
    params = [ module.weight, module.bias ]
    size = comfy.memory_management.vram_aligned_size(params)
    try:
        PIN_TOTAL += size
        #logging.info(f"PINNED {module.seed_key} for {PIN_TOTAL / (1024 ** 2)}MB total")
        module._pin = torch.empty((size,), dtype=torch.uint8, pin_memory=True)
    except:
        logging.warning(f"PIN failed for weight {module.seed_key}")
        module.pin_failed = True
        return False
    comfy.model_management.cast_to_gathered(params, module._pin)
    return True

def unpin_memory(module):
    global PIN_TOTAL
    if get_pin(module) is not None:
        PIN_TOTAL -= module._pin.numel()
        #logging.info(f"UNPINNED {module.seed_key} for {PIN_TOTAL / (1024 ** 2)}MB total")
        del module._pin


def vram_aligned_size(tensor):
    if isinstance(tensor, list):
        return sum([vram_aligned_size(t) for t in tensor])

    if isinstance(tensor, QuantizedTensor):
        inner_tensors, _ = tensor.__tensor_flatten__()
        return vram_aligned_size([ getattr(tensor, attr) for attr in inner_tensors ])

    if tensor is None:
        return 0

    size = tensor.numel() * tensor.element_size()
    aligment_req = 1024
    return (size + aligment_req - 1) // aligment_req * aligment_req

def interpret_gathered_like(tensors, r):
    r_offset = 0
    dest_views = []

    for tensor in tensors:

        if tensor is None:
            dest_views.append(None)
            continue

        if isinstance(tensor, QuantizedTensor):
            inner_tensors, qt_ctx = tensor.__tensor_flatten__()
            templates = { attr: getattr(tensor, attr) for attr in inner_tensors }
        else:
            templates = { "data": tensor }

        actuals = {}
        for attr, template in templates.items():
            size = template.numel() * template.element_size()
            actuals[attr] = r[r_offset:r_offset+size].view(dtype=template.dtype).view(template.shape)
            r_offset += vram_aligned_size(template)

        if isinstance(tensor, QuantizedTensor):
            dest_views.append(QuantizedTensor.__tensor_unflatten__(actuals, qt_ctx, 0, 0))
        else:
            dest_views.append(actuals["data"])

    return dest_views


def get_tensor_from_raw_ptr(ptr, size, device):
    container = {
        "shape": (size,),
        "typestr": "|u1",
        "data": (ptr, False), #writable
        "version": 3,
    }
            
    class Holder:
        pass
        
    holder = Holder() 
    holder.__cuda_array_interface__ = container
    
    return torch.as_tensor(holder, device=device)

def aimdo_to_tensor(alloc, device):
    _, ptr, size = alloc
    return get_tensor_from_raw_ptr(ptr, size, device)

#pytorch doesnt have an API for a CUDAPluggableAllocator from an already loaded
#library. Rather than force a second load that pytorch owns, construct these
#pytorch internals outselves as sperate CDLL loads is far too risky.

class CUDAPluggableAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self, lib, alloc_fn_name: str, free_fn_name: str):
        alloc_fn = ctypes.cast(getattr(lib, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(lib, free_fn_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)

aimdo_allocator = CUDAPluggableAllocator(aimdo.control.lib, "alloc_fn", "free_fn")
