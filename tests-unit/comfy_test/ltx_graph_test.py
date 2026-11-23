from contextlib import contextmanager, nullcontext
from unittest.mock import Mock
import weakref

import pytest
import torch

import comfy.model_prefetch as prefetch
from comfy.ldm.lightricks.model import LTXVModel, graph_input_key
from comfy.ldm.lightricks.av_model import LTXAVModel


def test_graph_input_key_tracks_storage_and_layout():
    x = torch.empty(2, 3)
    assert graph_input_key((x, None)) == graph_input_key((x.view_as(x), None))
    assert graph_input_key((x,)) != graph_input_key((x.clone(),))
    assert graph_input_key((x,)) != graph_input_key((x.t(),))


@pytest.mark.parametrize("old_key,new_key", [("same", "same"), ("old", "new"), (None, None)])
def test_prefetch_replay_checks_input_key(monkeypatch, old_key, new_key):
    graph = Mock()
    module = Mock(_v_block=object(), _comfy_graph={"graph": graph, "signature": 7, "key": old_key})
    malloc = Mock(_comfy_active=True)
    malloc.iterate.return_value = False
    malloc.use_stream.return_value = nullcontext()
    monkeypatch.setitem(prefetch.MALLOC_GRAPHS, prefetch.threading.get_ident(), malloc)
    monkeypatch.setattr(prefetch.args, "disable_cuda_graphs", False)
    monkeypatch.setattr(prefetch.comfy.model_management, "is_device_cuda", lambda device: True)
    monkeypatch.setitem(prefetch.GRAPH_CAPTURE_STREAMS, "cuda", Mock())
    monkeypatch.setattr(prefetch.comfy_aimdo.model_vbar, "vbar_fault", lambda block: 7)
    monkeypatch.setattr(prefetch.comfy_aimdo.model_vbar, "vbar_signature_compare", lambda a, b: a == b)
    core = Mock()
    if old_key == new_key:
        queue = [None, None]
        prefetch.prefetch_queue_pop(queue, "cuda", module, core=core, enable_graph=True, graph_key=new_key)
        graph.replay.assert_called_once()
        core.assert_not_called()
    else:
        queue = [None, None]
        monkeypatch.setattr(prefetch.torch.cuda, "stream", lambda stream: nullcontext())
        monkeypatch.setattr(prefetch.comfy.model_management, "current_stream", lambda device: Mock())
        monkeypatch.setattr(prefetch, "GRAPH_WARMED_MODULES", set())
        prefetch.prefetch_queue_pop(queue, "cuda", module, core=core, enable_graph=True, graph_key=new_key)
        graph.reset.assert_called_once()
        graph.replay.assert_not_called()
        core.assert_called_once()


@pytest.mark.parametrize("audio_video", [False, True])
@pytest.mark.parametrize("stg", [False, True])
def test_ltx_forward_scopes_preparation_and_not_outputs(monkeypatch, audio_video, stg):
    events = []
    prepared = []
    freed = []

    @contextmanager
    def scope(device):
        events.append("begin")
        yield
        assert all(reference() is None for reference in prepared)
        assert freed and all(event == "resume" for event in freed)
        events.append("end")

    @contextmanager
    def pause():
        events.append("pause")
        yield
        events.append("resume")

    monkeypatch.setattr(prefetch, "malloc_graph_scope", scope)
    monkeypatch.setattr(prefetch, "pause_malloc_graph", pause)
    cls = LTXAVModel if audio_video else LTXVModel
    model = object.__new__(cls)
    torch.nn.Module.__init__(model)

    def prepare(x, *args, **kwargs):
        assert events == ([] if stg else ["begin"])
        values = [value.clone() for value in x] if audio_video else [x.clone()]
        prepared.extend(weakref.ref(value, lambda reference: freed.append(events[-1])) for value in values)
        return values if audio_video else values[0], None, {}

    model._process_input = prepare
    model._prepare_timestep = lambda *args, **kwargs: (None, None, None)
    model._prepare_context = lambda context, batch, x, mask: (context, mask)
    model._prepare_attention_mask = lambda mask, dtype: mask
    model._prepare_positional_embeddings = lambda *args: None
    model._build_guide_self_attention_mask = lambda *args: None
    model._process_transformer_blocks = lambda x, *args, **kwargs: x

    def output(x, *args, **kwargs):
        assert events == (["pause"] if stg else ["begin", "pause"])
        return [value.clone() for value in x] if audio_video else x.clone()

    model._process_output = output
    x = torch.empty(1, 2, 3)
    result = model([x, x] if audio_video else x, torch.ones(1), x, attention_mask=None,
                   transformer_options={"stg_self_attn_blocks": (0,)} if stg else {})
    assert torch.equal(result[0] if audio_video else result, x)
    assert events == (["pause", "resume"] if stg else ["begin", "pause", "resume", "end"])
    assert all(reference() is None for reference in prepared)
