import torch
import logging
import comfy.model_management
import comfy.memory_management

from comfy.cli_args import args

def get_pin(module):
    return getattr(module, "_pin", None)

ALL_PINS=[]

def pin_memory(module):
    global ALL_PINS
    if module.pin_failed or args.disable_pinned_memory or get_pin(module) is not None:
        return
    #FIXME: This is a RAM cache trigger event
    params = [ module.weight, module.bias ]
    size = comfy.memory_management.vram_aligned_size(params)
    try:
        pin = torch.empty((size,), dtype=torch.uint8)
        if comfy.model_management.pin_memory(pin):
            module._pin = pin
            ALL_PINS.append(module)
        else:
            logging.warning(f"PIN failed for weight {module.seed_key}")
            module.pin_failed = True
            return False
    except:
        logging.warning(f"PIN failed for weight {module.seed_key}")
        module.pin_failed = True
        return False
    return True

def unpin_memory(module):
    if get_pin(module) is not None:
        comfy.model_management.unpin_memory(module._pin)
        del module._pin

def unpin_all():
    for module in ALL_PINS:
        unpin_memory(module)
    ALL_PINS.clear()


