import torch

import comfy_aimdo.control


_graphs = {}


def _signature(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    if isinstance(value, dict):
        items = ((k, _signature(v)) for k, v in value.items())
        return ("dict", tuple(sorted(items, key=lambda item: repr(item[0]))))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_signature(v) for v in value))
    if isinstance(value, (set, frozenset)):
        return (type(value), tuple(sorted((_signature(v) for v in value), key=repr)))
    return (type(value), id(value))


def key(*values):
    return tuple(_signature(value) for value in values)


def get(model, graph_key):
    cached = _graphs.get(model)
    if cached is None:
        return False, None
    cached_key, graph = cached
    if cached_key == graph_key:
        del _graphs[model]
        return True, graph
    if graph is not None:
        comfy_aimdo.control.destroy_record(graph)
    del _graphs[model]
    return False, None


def put(model, graph_key, graph):
    cached = _graphs.get(model)
    if cached is not None and cached[1] is not None and cached[1] != graph:
        comfy_aimdo.control.destroy_record(cached[1])
    _graphs[model] = graph_key, graph


def clear(model):
    cached = _graphs.get(model)
    if cached is not None and cached[1] is not None:
        comfy_aimdo.control.destroy_record(cached[1])
    _graphs.pop(model, None)
