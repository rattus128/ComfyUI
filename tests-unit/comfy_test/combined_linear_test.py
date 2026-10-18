import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.lora  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.model_prefetch  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402


def test_combined_linear_declines_without_running_split_or_activation():
    linears = [comfy.ops.manual_cast.Linear(16, n, bias=False) for n in (24, 40)]
    combined = comfy.ops.CombinedLinear(linears)
    x = torch.randn(2, 16)
    original = x.clone()
    assert combined.forward(x) is NotImplemented
    assert comfy.ops.linear_input_act(combined, x, "model_owned_activation") is NotImplemented
    assert torch.equal(x, original)


def test_resident_int8_lora_relayout_only_on_signature_hits():
    if not comfy.memory_management.aimdo_enabled or not torch.cuda.is_available():
        pytest.skip("requires initialized DynamicVRAM")

    torch.manual_seed(53)
    device = torch.device("cuda:0")
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)

    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = ops.Linear(512, 96, bias=False)
            self.b = ops.Linear(512, 160, bias=False)
            for layer in (self.a, self.b):
                weight = QuantizedTensor.from_float(
                    torch.randn(layer.out_features, 512, dtype=torch.bfloat16) * 0.03,
                    "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=256,
                )
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                layer.quant_format = "int8_tensorwise"
                layer.layout_type = weight.layout_cls
            self.combined = comfy.ops.CombinedLinear((self.a, self.b))

        def get_dynamic_vram__units(self):
            return [self], []

        def forward(self, x):
            ab = self.combined.forward(x)
            return torch.cat((self.a(x), self.b(x)), dim=-1) if ab is NotImplemented else ab

    model = Projections()
    members = (model.a, model.b)
    base = comfy.model_patcher.ModelPatcherDynamic(model, device, torch.device("cpu"))
    patches = comfy.lora.load_lora({
        "a.lora_up.weight": torch.randn(96, 4) * 0.1,
        "a.lora_down.weight": torch.randn(4, 512) * 0.1,
    }, {"a": "a.weight"})
    positive, negative = base.clone(), base.clone()
    positive.add_patches(patches, strength_patch=0.75)
    negative.add_patches(patches, strength_patch=-0.5)
    x = torch.randn(3, 512, device=device, dtype=torch.bfloat16)
    results = []
    try:
        for patcher in (base, positive, negative, base):
            patcher.partially_load(device)
            assert model.combined.weight is None
            comfy.model_prefetch.pin_modules(members, device, x.dtype)
            try:
                cold = model(x)
                canonical = [s._v_weight.dequantize().clone() for s in members]
                model.combined.prepare(device)
                assert model.combined.weight is None
                assert torch.equal(cold, model(x))
            finally:
                comfy.model_prefetch.cleanup_prefetched_modules(model, members)

            comfy.model_prefetch.pin_modules(members, device, x.dtype)
            try:
                assert model.combined.weight is not None
                hot = model(x)
                for s, expected in zip(members, canonical):
                    assert torch.equal(s._v_weight.dequantize(), expected)
                assert torch.equal(hot, cold)
                results.append(hot.clone())
            finally:
                comfy.model_prefetch.cleanup_prefetched_modules(model, members)
        assert torch.equal(results[0], results[3])
        assert not torch.equal(results[0], results[1])
        assert not torch.equal(results[1], results[2])
    finally:
        torch.cuda.synchronize()
        base.unpatch_model(unpatch_weights=True)
