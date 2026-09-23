import logging
import os
import threading
import warnings
import weakref

import comfy_kitchen as ck
import torch

import comfy_aimdo.malloc_graph
import comfy_aimdo.model_vbar
from comfy.cli_args import args
import comfy.memory_management
import comfy.model_management
import comfy.ops

PREFETCH_QUEUES = []
GRAPH_WARMED_MODULES = weakref.WeakSet()
GRAPH_CAPTURE_STREAMS = {}
MALLOC_GRAPHS = {}
MALLOC_GRAPH_BREAKS = 0
MALLOC_GRAPH_ROGUES = 0
MALLOC_GRAPH_USED = False
# read-order entries per step; the streamed W4A8 GEMM records K/(32*PackRows) runs per weight
PREFETCH_RING_CAPACITY = 32768
# A/B test scaffold: COMFY_PREFETCH_RING_MIB overrides the ring lookahead
PREFETCH_RING_LOOKAHEAD = int(float(os.environ.get("COMFY_PREFETCH_RING_MIB", "8")) * 1024 * 1024)
# diagnostics: log the issuer's cumulative counters every N steps (synchronizes the device)
PREFETCH_RING_STATS_EVERY = int(os.environ.get("COMFY_PREFETCH_RING_STATS_EVERY", "0"))
PREFETCH_RING_CHUNK = 96 * 1024
ACTIVE_PREFETCH_RING = None
PREFETCH_RING_MODULES = weakref.WeakSet()


class CompiledPrefetchRing:
    def __init__(self, device, cache_key):
        self.device = device
        self.cache_key = cache_key
        self.entries = []
        self.ready = False
        with pause_malloc_graph(sync=True):
            self.descriptors = torch.empty(
                (PREFETCH_RING_CAPACITY, 2), device=device, dtype=torch.uint64
            )

    def record(self, tensor):
        if not tensor.is_cuda or tensor.device != self.descriptors.device:
            raise RuntimeError("prefetch ring region must be on the ring CUDA device")
        if not tensor.is_contiguous():
            raise RuntimeError("prefetch ring region must be contiguous")
        if len(self.entries) == PREFETCH_RING_CAPACITY:
            raise RuntimeError("prefetch ring descriptor capacity exceeded")
        size = tensor.numel() * tensor.element_size()
        if tensor.data_ptr() % 16 or size % 16:
            raise RuntimeError("prefetch ring region must be 16-byte aligned (bulk prefetch granule)")
        if not size:
            return
        if self.entries and self.entries[-1][0] + self.entries[-1][1] == tensor.data_ptr():
            self.entries[-1] = (self.entries[-1][0], self.entries[-1][1] + size)
        else:
            self.entries.append((tensor.data_ptr(), size))

    def finish_recording(self):
        ck.set_prefetch_ring_recorder(None)
        if not self.entries:
            return
        with pause_malloc_graph(sync=True):
            host = torch.tensor(self.entries, dtype=torch.uint64)
            self.descriptors[:len(self.entries)].copy_(host)
        self.ready = True
        logging.info(
            "Comfy prefetch ring recorded %d regions (%.2f GiB)",
            len(self.entries), sum(size for _, size in self.entries) / (1024 ** 3),
        )

    def configure(self):
        ck.configure_prefetch_ring(
            self.descriptors, len(self.entries), PREFETCH_RING_LOOKAHEAD, PREFETCH_RING_CHUNK
        )


def _prefetch_ring_cache_key(past_key_values):
    key = []
    for cache in past_key_values:
        for name in ("key", "value", "recurrent_state", "conv_state"):
            tensor = getattr(cache, name, None)
            if tensor is not None:
                key.append((tensor.data_ptr(), tensor.numel(), tensor.element_size()))
    return tuple(key)


def prefetch_ring_begin(module, device, past_key_values, enabled):
    global ACTIVE_PREFETCH_RING
    if not enabled or not hasattr(ck, "prefetch_ring_is_available") or not ck.prefetch_ring_is_available():
        return None
    cache_key = _prefetch_ring_cache_key(past_key_values)
    ring = getattr(module, "_compiled_prefetch_ring", None)
    if ring is None or ring.cache_key != cache_key:
        ring = CompiledPrefetchRing(device, cache_key)
        module._compiled_prefetch_ring = ring
        PREFETCH_RING_MODULES.add(module)
    if ring.ready:
        if ACTIVE_PREFETCH_RING is not ring:
            ring.configure()
            ACTIVE_PREFETCH_RING = ring
        if PREFETCH_RING_STATS_EVERY:
            ring.steps = getattr(ring, "steps", 0) + 1
            if ring.steps % PREFETCH_RING_STATS_EVERY == 0:
                total, consumed, stalled, touched, skipped, waited_ns, smids, distinct = ck.prefetch_ring.stats()
                logging.info(
                    "Comfy prefetch ring after %d steps: touched %.2f GB skipped %.2f GB waited %.1f ms/step stalled %d smids %s distinct-SM hist %s",
                    ring.steps, touched / 1e9, skipped / 1e9, waited_ns / 1e6 / ring.steps, stalled, smids, distinct,
                )
        ck.start_prefetch_ring(device)
    else:
        if ACTIVE_PREFETCH_RING is not None:
            ck.disable_prefetch_ring(ACTIVE_PREFETCH_RING.device)
            ACTIVE_PREFETCH_RING = None
        ck.set_prefetch_ring_recorder(ring.record)
    return ring


def prefetch_ring_end(ring):
    if ring is None:
        return
    if not ring.ready:
        ring.finish_recording()

def _malloc_graph_break():
    global MALLOC_GRAPH_BREAKS
    MALLOC_GRAPH_BREAKS += 1
    logging.debug("Comfy model compiler graph break")

def malloc_graph_enabled(device):
    return not args.disable_comfy_compiler and comfy.memory_management.aimdo_enabled and comfy.model_management.is_device_cuda(device)

class _PauseMallocGraph:
    def __init__(self, sync=False):
        self.sync = sync

    def __enter__(self):
        graph = MALLOC_GRAPHS.get(threading.get_ident())
        if graph is not None and graph._comfy_active:
            graph.pause(sync=self.sync)

    def __exit__(self, *args):
        graph = MALLOC_GRAPHS.get(threading.get_ident())
        if graph is not None and graph._comfy_active:
            graph.resume(sync=self.sync)

def pause_malloc_graph(sync=False):
    return _PauseMallocGraph(sync)

class _MallocGraphScope:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        malloc_graph_begin(self.device)

    def __exit__(self, exc_type, *args):
        if exc_type is None:
            malloc_graph_end()
        else:
            cleanup_malloc_graph()

def malloc_graph_scope(device):
    return _MallocGraphScope(device)

def malloc_graph_begin(device):
    global MALLOC_GRAPH_USED
    if not malloc_graph_enabled(device):
        return
    thread_id = threading.get_ident()
    graph = MALLOC_GRAPHS.get(thread_id)
    if graph is None:
        graph = comfy_aimdo.malloc_graph.record(
            comfy.model_management.current_stream(device), args.assert_graph_breaks
        )
        graph._comfy_cuda_graph_modules = weakref.WeakSet()
        MALLOC_GRAPHS[thread_id] = graph
    else:
        graph.push()
    if hasattr(ck, "set_allocation_context"):
        ck.set_allocation_context(pause_malloc_graph())
    graph._comfy_active = True
    MALLOC_GRAPH_USED = True

def malloc_graph_end():
    thread_id = threading.get_ident()
    graph = MALLOC_GRAPHS.get(thread_id)
    if graph is not None and graph._comfy_active:
        if graph.pop():
            _malloc_graph_break()
        graph._comfy_active = False

def cleanup_malloc_graph():
    global MALLOC_GRAPH_ROGUES

    graph = MALLOC_GRAPHS.pop(threading.get_ident(), None)
    if graph is not None:
        if graph._comfy_active:
            graph.abort()
            graph._comfy_active = False
        for module in graph._comfy_cuda_graph_modules:
            _drop_graph(module)
        MALLOC_GRAPH_ROGUES += graph.rogue_count
        del graph

def pin_modules(comfy_modules, device, dtype=None):
    registerable_size = 0
    for s in comfy_modules:
        registerable_size += comfy.memory_management.vram_aligned_size([s.weight, s.bias])
        for param_key in ("weight", "bias"):
            lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
            if lowvram_fn is not None:
                registerable_size += lowvram_fn.memory_required()

    offload_stream, fully_faulted = comfy.ops.cast_modules_with_vbar(comfy_modules, None, device, None, True, return_faulted=True)
    if not (comfy_modules and comfy_modules[0]._pin_state["fast_disk"]):
        comfy.model_management.ensure_pin_registerable(registerable_size)
    comfy.model_management.sync_stream(device, offload_stream)
    if fully_faulted and dtype is not None:
        for comfy_module in comfy_modules:
            comfy.ops.resolve_cast_module_with_vbar(comfy_module, dtype, device, dtype, None, False, return_weights=False)
    return offload_stream, fully_faulted

def cleanup_prefetched_modules(module, comfy_modules):
    for s in comfy_modules:
        prefetch = getattr(s, "_prefetch", None)
        if prefetch is None:
            continue
        for param_key in ("weight", "bias"):
            lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
            if lowvram_fn is not None:
                lowvram_fn.clear_prepared()
        if prefetch["signature"] is not None:
            comfy_aimdo.model_vbar.vbar_unpin(s._v)
        delattr(s, "_prefetch")
    if getattr(module, "_v_block_faulted", False):
        comfy_aimdo.model_vbar.vbar_unpin(module._v_block)
        del module._v_block_faulted

def _drop_graph(module):
    graph = getattr(module, "_comfy_graph", None)
    if graph is None:
        return
    # reset() through the bound method surfaces the allocator's benign
    # "uncaptured free of a captured allocation" as catchable Python warnings;
    # a plain del frees from the C++ dealloc path and spams stderr instead
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph["graph"].reset()
    del module._comfy_graph

def cleanup_prefetch_queues():
    global PREFETCH_QUEUES
    global MALLOC_GRAPH_BREAKS
    global MALLOC_GRAPH_ROGUES
    global MALLOC_GRAPH_USED
    global ACTIVE_PREFETCH_RING

    if hasattr(ck, "set_prefetch_ring_recorder"):
        ck.set_prefetch_ring_recorder(None)
    if ACTIVE_PREFETCH_RING is not None and ACTIVE_PREFETCH_RING.ready:
        ck.disable_prefetch_ring(ACTIVE_PREFETCH_RING.device)
    ACTIVE_PREFETCH_RING = None
    if PREFETCH_RING_MODULES:
        comfy.model_management.synchronize()
    cleanup_malloc_graph()
    for queue in PREFETCH_QUEUES:
        for entry in queue:
            if entry is None or not isinstance(entry, tuple):
                continue
            _, prefetch_state = entry
            prefetched_module, comfy_modules = prefetch_state
            if comfy_modules is not None:
                cleanup_prefetched_modules(prefetched_module, comfy_modules)
    for module in PREFETCH_RING_MODULES:
        if hasattr(module, "_compiled_prefetch_ring"):
            del module._compiled_prefetch_ring
    PREFETCH_QUEUES = []
    GRAPH_WARMED_MODULES.clear()
    if MALLOC_GRAPH_USED:
        logging.info("Comfy model compiler graph breaks: %d, rogues: %d", MALLOC_GRAPH_BREAKS, MALLOC_GRAPH_ROGUES)
    MALLOC_GRAPH_BREAKS = 0
    MALLOC_GRAPH_ROGUES = 0
    MALLOC_GRAPH_USED = False

def prefetch_queue_pop(queue, device, module, dtype=None, core=None, enable_graph=False, generator=None, malloc_scope=None):
    malloc_graph = MALLOC_GRAPHS.get(threading.get_ident())
    if malloc_graph is not None and not malloc_graph._comfy_active:
        malloc_graph = None
    enable_graph = enable_graph and malloc_graph is not None and not args.disable_cuda_graphs and comfy.model_management.is_device_cuda(device) and getattr(module, "_v_block", None) is not None
    if queue is None:
        if malloc_graph is not None and malloc_scope is not None:
            if malloc_graph.iterate(malloc_scope if module is not None else None):
                _malloc_graph_break()
        if core is not None:
            core()
        return

    capture_stream = None
    if enable_graph:
        capture_stream = GRAPH_CAPTURE_STREAMS.get(device)
        if capture_stream is None:
            capture_stream = torch.cuda.Stream(device=device)
            # Keep PyTorch's persistent BLAS workspaces outside the allocation graph.
            malloc_graph.pause()
            with torch.cuda.stream(capture_stream):
                torch.cuda.current_blas_handle()
                one = torch.empty((2, 2), device=device)
                torch.addmm(one[0], one, one)
            malloc_graph.resume()
            GRAPH_CAPTURE_STREAMS[device] = capture_stream

    signature = None
    graph_hit = False
    graph = getattr(module, "_comfy_graph", None) if enable_graph else None
    if graph is not None:
        signature = comfy_aimdo.model_vbar.vbar_fault(module._v_block)
        if signature is not None:
            module._v_block_faulted = True
            graph_hit = comfy_aimdo.model_vbar.vbar_signature_compare(signature, graph["signature"])

    if malloc_graph is not None and malloc_scope is not None:
        if malloc_graph.iterate(malloc_scope if module is not None and not graph_hit else None):
            _malloc_graph_break()

    consumed = queue.pop(0)
    if consumed is not None:
        offload_stream, prefetch_state = consumed
        if offload_stream is not None:
            offload_stream.wait_stream(comfy.model_management.current_stream(device))
        prefetched_module, comfy_modules = prefetch_state
        if comfy_modules is not None:
            cleanup_prefetched_modules(prefetched_module, comfy_modules)

    if graph_hit:
        queue[0] = (None, (module, []))
        graph["graph"].replay()
        return

    fully_faulted = False
    prefetch = queue[0]
    if prefetch is not None:
        comfy_modules = []
        prefetch_modules = prefetch if isinstance(prefetch, (list, tuple)) else (prefetch,)
        for root in prefetch_modules:
            for s in root.modules():
                if hasattr(s, "_v"):
                    comfy_modules.append(s)

        offload_stream, fully_faulted = pin_modules(comfy_modules, device, dtype)
        queue[0] = (offload_stream, (module, comfy_modules))

    if core is not None:
        if enable_graph and fully_faulted and module in GRAPH_WARMED_MODULES:
            if signature is None:
                signature = comfy_aimdo.model_vbar.vbar_fault(module._v_block)
                if signature is not None:
                    module._v_block_faulted = True
            if signature is not None:
                _drop_graph(module)
                malloc_graph.pause()
                graph = torch.cuda.CUDAGraph()
                if generator is not None:
                    graph.register_generator_state(generator)
                malloc_graph.resume()
                # Capture-time VBAR eviction is safe after prior work completes.
                # The device sync would deadlock against a polling ring issuer
                # waiting on this stream, so stop it for the rest of the step.
                if ACTIVE_PREFETCH_RING is not None:
                    ck.disable_prefetch_ring(device)
                comfy.model_management.synchronize()
                capture_stream.wait_stream(comfy.model_management.current_stream(device))
                malloc_graph.pause(sync=True)
                with malloc_graph.use_stream(capture_stream):
                    with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                        malloc_graph.resume()
                        core()
                        malloc_graph.pause()
                malloc_graph.resume(sync=True)
                comfy.model_management.current_stream(device).wait_stream(capture_stream)
                graph.replay()
                module._comfy_graph = {"graph": graph, "signature": signature}
                malloc_graph._comfy_cuda_graph_modules.add(module)
                return
        if capture_stream is None:
            core()
        else:
            capture_stream.wait_stream(comfy.model_management.current_stream(device))
            with torch.cuda.stream(capture_stream), malloc_graph.use_stream(capture_stream):
                core()
            comfy.model_management.current_stream(device).wait_stream(capture_stream)
            GRAPH_WARMED_MODULES.add(module)

def make_prefetch_queue(queue, device, transformer_options):
    if (not transformer_options.get("prefetch_dynamic_vbars", False)
        or comfy.model_management.NUM_STREAMS == 0
        or comfy.model_management.is_device_cpu(device)
        or not comfy.model_management.device_supports_non_blocking(device)):
        return None

    queue = [None] + queue + [None]
    PREFETCH_QUEUES.append(queue)
    return queue
