import logging
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
from comfy.quant_ops import QuantizedTensor

PREFETCH_QUEUES = []
GRAPH_WARMED_MODULES = weakref.WeakSet()
GRAPH_CAPTURE_STREAMS = {}
MALLOC_GRAPHS = {}
MALLOC_GRAPH_BREAKS = 0
MALLOC_GRAPH_ROGUES = 0
MALLOC_GRAPH_USED = False
# Prefetch ring (lookahead MiB, min lead MiB) by device L2 MiB, measured on the RTX 5090 and RTX 5080.
# Overshooting the L2 evicts prefetched weights before their GEMV reads them; the ring stays off on
# unmeasured parts (the 24 MiB RTX 5060 lost 12% at half its L2). min_lead is the in-flight depth:
# a chunk requested closer than that to the consumed position lands after its reader dropped it.
PREFETCH_RING_LOOKAHEAD_MIB = {96: (60, 16), 64: (55, 8)}
PREFETCH_RING_CHUNK = 96 * 1024
# Bumped whenever a graphed layer runs outside graph replay or an execution ends: the resident
# weight addresses a ring was built from may have moved.
PREFETCH_RING_GENERATION = 0
ACTIVE_PREFETCH_RING = None
PREFETCH_RING_MODULES = weakref.WeakSet()


def _prefetch_ring_credit(cache):
    # attention caches whose reading kernels credit the ring: flash decode K/V rows, DeltaNet state
    return {"kv": ck.prefetch_ring.CREDIT_KV, "delta": ck.prefetch_ring.CREDIT_DELTA}.get(cache.prefetch_credit, 0)


def _prefetch_ring_regions(layers, caches, seq_len):
    """(address, bytes, flags) of everything the step's decode kernels read, in read order, the
    KV region indices and the credit mask. A layer lists its reads through prefetch_reads(cache)
    (its cache, ops modules and stable plain tensors), otherwise its quantized weights in module
    order. A layer's attention state is listed ahead of its weights: a late credit only narrows
    the window, an early one would let the issuer skip weights. Reads no decode kernel credits
    are flagged for the issuer to credit itself. None while a weight is not VBAR-resident."""
    regions = []
    kv_rows = []
    credits = 0

    def add(address, size, flags=0):
        if address % 16 == 0 and size % 16 == 0:  # bulk prefetch granularity
            regions.append((address, size, flags))

    def add_tensor(tensor):
        add(tensor.data_ptr(), tensor.numel() * tensor.element_size(), ck.prefetch_ring.SELF_CREDIT)

    def add_module(s):
        weight = getattr(s, "_v_weight", None)
        if weight is None:
            return not isinstance(s.weight, QuantizedTensor)
        if isinstance(weight, QuantizedTensor):
            # only the streamed W4A8 decode and the int8 GEMV credit the ring
            if weight._layout_cls == "AsymW4A8Int8Layout":
                if not weight._params.mma_packed or seq_len > 8:
                    return True
            elif weight._layout_cls != "TensorWiseINT8Layout" or seq_len > 2 or weight.shape[1] % 16:
                return True
            add(weight._qdata.data_ptr(), weight._qdata.numel() * weight._qdata.element_size())
        else:
            add_tensor(weight)
        if s._v_bias is not None:
            add_tensor(s._v_bias)
        return True

    def add_cache(cache):
        nonlocal credits
        credit = _prefetch_ring_credit(cache)
        state = cache.prefetch_regions(seq_len) if credit else []
        if state:
            credits |= credit
        if credit == ck.prefetch_ring.CREDIT_KV:
            kv_rows.extend(range(len(regions), len(regions) + len(state)))
        regions.extend((address, size, 0) for address, size in state)

    for layer, cache in zip(layers, caches):
        if hasattr(layer, "prefetch_reads"):
            reads = layer.prefetch_reads(cache)
        else:
            reads = [cache] + [s for s in layer.modules() if isinstance(getattr(s, "weight", None), QuantizedTensor)]
        for read in reads:
            if read is None:
                continue
            if isinstance(read, torch.Tensor):
                add_tensor(read)
            elif hasattr(read, "prefetch_credit"):
                add_cache(read)
            elif not add_module(read):
                return None, None, 0
    return regions, kv_rows, credits


class CompiledPrefetchRing:
    def __init__(self, device, key, regions, kv_rows, credits):
        self.device = device
        self.key = key
        self.credits = credits
        # KV byte counts grow every step: rewritten in a pinned double buffer and copied over the
        # device list before the issuer starts (start() re-sums the step total)
        self.kv_rows = None
        with pause_malloc_graph(sync=True):
            self.descriptors = torch.tensor(regions, dtype=torch.int64, device=device)
            if kv_rows:
                self.kv_rows = torch.tensor(kv_rows, dtype=torch.int64)
                self.host = torch.tensor([regions, regions], dtype=torch.int64).pin_memory()
                self.copied = [torch.cuda.Event(), torch.cuda.Event()]
                self.parity = 0
        self.lookahead, self.min_lead = prefetch_ring_lookahead(device)
        logging.info("Comfy prefetch ring: %d regions (%.2f GiB), lookahead %d MiB", len(regions),
                     sum(size for _, size, _ in regions) / (1024 ** 3), self.lookahead >> 20)

    def configure(self):
        ck.configure_prefetch_ring(self.descriptors.view(torch.uint64), len(self.descriptors), self.lookahead, self.min_lead, PREFETCH_RING_CHUNK, self.credits)

    def refresh(self, kv_bytes):
        self.parity ^= 1
        # the pinned row is reused every other step: its previous copy must have landed
        self.copied[self.parity].synchronize()
        host = self.host[self.parity]
        host[:, 1].index_fill_(0, self.kv_rows, kv_bytes)
        self.descriptors.copy_(host, non_blocking=True)
        self.copied[self.parity].record(torch.cuda.current_stream(self.device))


def _stop_prefetch_ring():
    global ACTIVE_PREFETCH_RING
    if ACTIVE_PREFETCH_RING is not None:
        ck.disable_prefetch_ring(ACTIVE_PREFETCH_RING.device)
        ACTIVE_PREFETCH_RING = None


def prefetch_ring_lookahead(device):
    lookahead, min_lead = PREFETCH_RING_LOOKAHEAD_MIB.get(torch.cuda.get_device_properties(device).L2_cache_size >> 20, (0, 0))
    return lookahead << 20, min_lead << 20


def prefetch_ring_begin(layers, device, seq_len, caches):
    """Start the L2 prefetch ring over one graphed decode step of `layers`."""
    global ACTIVE_PREFETCH_RING
    malloc_graph = MALLOC_GRAPHS.get(threading.get_ident())
    # eager layers do not bump the generation: their weights may move under a built ring
    if malloc_graph is None or not malloc_graph._comfy_active or args.disable_cuda_graphs or not ck.prefetch_ring_is_available():
        _stop_prefetch_ring()
        return
    key = (PREFETCH_RING_GENERATION, seq_len)
    ring = getattr(layers, "_prefetch_ring", None)
    if ring is None or ring.key != key:
        # build only after a full step at this generation (every layer replayed its graph), so
        # the resident weights are where _v_weight says they are
        regions = None
        if getattr(layers, "_prefetch_ring_seen", None) == key and prefetch_ring_lookahead(device)[0] > 0:
            regions, kv_rows, credits = _prefetch_ring_regions(layers, caches, seq_len)
        layers._prefetch_ring_seen = key
        if not regions:
            _stop_prefetch_ring()
            return
        ring = CompiledPrefetchRing(device, key, regions, kv_rows, credits)
        layers._prefetch_ring = ring
        PREFETCH_RING_MODULES.add(layers)
    if ACTIVE_PREFETCH_RING is not ring:
        ring.configure()
        ACTIVE_PREFETCH_RING = ring
    if ring.kv_rows is not None:
        # every layer's cache advances in lockstep, so one cache's row count serves all KV regions
        ring.refresh(next(c for c in caches if c.prefetch_credit == "kv").prefetch_regions(seq_len)[0][1])
    ck.start_prefetch_ring(device)

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
        else:
            comfy.model_management.synchronize()
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
    if dtype is not None:
        groups = dict.fromkeys(s._combined_linear for s in comfy_modules if hasattr(s, "_combined_linear"))
        for group in groups:
            group.prepare(device)
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
    global PREFETCH_RING_GENERATION

    _stop_prefetch_ring()
    PREFETCH_RING_GENERATION += 1
    if PREFETCH_RING_MODULES:
        comfy.model_management.synchronize()
    cleanup_malloc_graph()
    for queue in PREFETCH_QUEUES:
        # Only the head has been prefetched; later entries are raw module groups.
        if queue[0] is not None:
            _, (prefetched_module, comfy_modules) = queue[0]
            cleanup_prefetched_modules(prefetched_module, comfy_modules)
    for module in PREFETCH_RING_MODULES:
        del module._prefetch_ring
    PREFETCH_RING_MODULES.clear()
    PREFETCH_QUEUES = []
    GRAPH_WARMED_MODULES.clear()
    if MALLOC_GRAPH_USED:
        logging.info("Comfy model compiler graph breaks: %d, rogues: %d", MALLOC_GRAPH_BREAKS, MALLOC_GRAPH_ROGUES)
    MALLOC_GRAPH_BREAKS = 0
    MALLOC_GRAPH_ROGUES = 0
    MALLOC_GRAPH_USED = False

def prefetch_queue_pop(queue, device, module, dtype=None, core=None, enable_graph=False, generator=None, malloc_scope=None):
    global PREFETCH_RING_GENERATION
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
            graph_hit = graph_hit and all(group.weight is weight for group, weight in graph["combined_weights"])

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
    if enable_graph and module is not None:
        # outside replay the layer's weights may land at new addresses, and a capture's device
        # sync would deadlock against a polling ring issuer
        PREFETCH_RING_GENERATION += 1
        _stop_prefetch_ring()

    fully_faulted = False
    prefetch = queue[0]
    if prefetch is not None:
        comfy_modules = []
        prefetch_modules = prefetch if isinstance(prefetch, (list, tuple)) else (prefetch,)
        for root in prefetch_modules:
            for s in root.modules():
                if hasattr(s, "_v"):
                    comfy_modules.append(s)

        # Keep partially pinned groups reachable if pinning raises.
        queue[0] = (None, (module, comfy_modules))
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
                groups = dict.fromkeys(s._combined_linear for s in comfy_modules if hasattr(s, "_combined_linear"))
                module._comfy_graph = {"graph": graph, "signature": signature,
                                       "combined_weights": [(group, group.weight) for group in groups]}
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
