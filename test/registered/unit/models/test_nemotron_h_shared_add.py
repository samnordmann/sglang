from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.layers.quantization.unquant import (
    Bf16GemmBackend,
    UnquantizedLinearMethod,
)
from sglang.srt.models.nemotron_h import NemotronHMoE
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _make_projection(input_size: int, output_size: int):
    weight = torch.randn(
        output_size,
        input_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) / (input_size**0.5)
    method = UnquantizedLinearMethod()
    return SimpleNamespace(weight=weight, bias=None, quant_method=method), method


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_latent_projection_accumulates_into_shared_output_and_replays_graph(
    monkeypatch,
):
    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", Bf16GemmBackend.CUTEDSL)
    monkeypatch.setattr(unquant, "_use_cutedsl_bf16_gemm", lambda *args: False)
    projection, _ = _make_projection(64, 128)
    moe = SimpleNamespace(fc2_latent_proj=projection)
    routed = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    reference = F.linear(routed, projection.weight) + shared

    candidate, remaining_shared = NemotronHMoE._apply_latent_projection(
        moe, routed, shared.clone()
    )
    assert remaining_shared is None
    torch.testing.assert_close(candidate, reference, rtol=1e-2, atol=3.125e-2)

    # The shared-expert producer rewrites this buffer on every real graph replay.
    graph_shared_input = shared.clone()
    NemotronHMoE._apply_latent_projection(moe, routed, graph_shared_input.clone())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_shared_output = graph_shared_input.clone()
        graph_output, graph_remaining_shared = NemotronHMoE._apply_latent_projection(
            moe, routed, graph_shared_output
        )
    graph.replay()
    torch.cuda.synchronize()

    assert graph_remaining_shared is None
    torch.testing.assert_close(graph_output, reference, rtol=1e-2, atol=3.125e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_latent_projection_preserves_fallback_for_noncontiguous_addend(monkeypatch):
    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", Bf16GemmBackend.TORCH)
    projection, method = _make_projection(64, 128)
    routed = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn(128, 4, device="cuda", dtype=torch.bfloat16).t()
    assert not shared.is_contiguous()

    candidate = method.apply_with_addend(projection, routed, shared)
    reference = F.linear(routed, projection.weight) + shared

    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    assert candidate.data_ptr() != shared.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_latent_projection_preserves_cutedsl_selected_path(monkeypatch):
    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", Bf16GemmBackend.CUTEDSL)
    monkeypatch.setattr(unquant, "_use_cutedsl_bf16_gemm", lambda *args: True)
    monkeypatch.setattr(
        unquant,
        "_cutedsl_bf16_gemm",
        lambda x, weight, bias: F.linear(x, weight, bias),
    )
    projection, method = _make_projection(64, 128)
    routed = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    shared_before = shared.clone()

    candidate = method.apply_with_addend(projection, routed, shared)
    reference = F.linear(routed, projection.weight) + shared

    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    torch.testing.assert_close(shared, shared_before, rtol=0, atol=0)
