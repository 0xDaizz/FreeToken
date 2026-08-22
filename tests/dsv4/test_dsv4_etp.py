"""DeepSeek-V4 expert-intermediate tensor parallel (ETP) invariants."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.distributed import DistributedInfo
from freetoken.engine.engine import _validate_dsv4_expert_partition
from freetoken.models.deepseek_v4.moe import MoE
from freetoken.models.deepseek_v4.weight import (
    _place_dsfp4,
    resolve_dsfp4_tp_partition,
)
from freetoken.moe.cpu_executor import (
    resolve_flag_coord_affinity,
    resolve_tp_threads_and_affinity,
)
from freetoken.moe.expert_banks import ExpertBanks
from torch import nn

H, I, E = 128, 256, 1


def _empty_rank_banks(local_i: int):
    return {
        "gate_up_packed": [torch.empty(E, 2 * local_i, H // 2, dtype=torch.uint8)],
        "gate_up_scale": [torch.empty(E, 2 * local_i, H // 32, dtype=torch.uint8)],
        "down_packed": [torch.empty(E, H, local_i // 2, dtype=torch.uint8)],
        "down_scale": [torch.empty(E, H, local_i // 32, dtype=torch.uint8)],
    }


def _u8(*shape, offset=0):
    return (
        (torch.arange(torch.tensor(shape).prod()).reshape(shape) + offset) % 251
    ).to(torch.uint8)


def test_partition_geometry_requires_128_aligned_local_width():
    args = SimpleNamespace(moe_inter_dim=2048)
    assert resolve_dsfp4_tp_partition(args, tp_rank=0, tp_size=2) == (0, 2, 1024)
    assert resolve_dsfp4_tp_partition(args, tp_rank=1, tp_size=2) == (1, 2, 1024)
    with pytest.raises(ValueError, match="divisible"):
        resolve_dsfp4_tp_partition(args, tp_rank=0, tp_size=3)
    with pytest.raises(ValueError, match="128"):
        resolve_dsfp4_tp_partition(
            SimpleNamespace(moe_inter_dim=384), tp_rank=0, tp_size=2
        )


def test_two_rank_bank_slices_are_complementary_and_reconstruct_full_expert():
    local_i = I // 2
    ranks = [_empty_rank_banks(local_i), _empty_rank_banks(local_i)]
    tensors = {
        "w1.weight": _u8(I, H // 2, offset=1),
        "w1.scale": _u8(I, H // 32, offset=2),
        "w3.weight": _u8(I, H // 2, offset=3),
        "w3.scale": _u8(I, H // 32, offset=4),
        "w2.weight": _u8(H, I // 2, offset=5),
        "w2.scale": _u8(H, I // 32, offset=6),
    }
    for rank, banks in enumerate(ranks):
        for suffix, tensor in tensors.items():
            _place_dsfp4(
                banks,
                f"layers.0.ffn.experts.0.{suffix}",
                tensor,
                I,
                tp_rank=rank,
                tp_size=2,
            )

    for name, suffix in (("gate_up_packed", "weight"), ("gate_up_scale", "scale")):
        w1 = torch.cat([ranks[0][name][0][0, :local_i], ranks[1][name][0][0, :local_i]])
        w3 = torch.cat([ranks[0][name][0][0, local_i:], ranks[1][name][0][0, local_i:]])
        assert torch.equal(w1, tensors[f"w1.{suffix}"])
        assert torch.equal(w3, tensors[f"w3.{suffix}"])
    assert torch.equal(
        torch.cat(
            [ranks[0]["down_packed"][0][0], ranks[1]["down_packed"][0][0]], dim=-1
        ),
        tensors["w2.weight"],
    )
    assert torch.equal(
        torch.cat([ranks[0]["down_scale"][0][0], ranks[1]["down_scale"][0][0]], dim=-1),
        tensors["w2.scale"],
    )

    per_rank = sum(
        t.numel() * t.element_size() for values in ranks[0].values() for t in values
    )
    full = sum(t.numel() * t.element_size() for t in tensors.values())
    assert per_rank * 2 == full


def _bundle(rank=0, size=2, local_i=I // 2):
    sources = _empty_rank_banks(local_i)
    return ExpertBanks(
        "ds_fp4",
        sources,
        expert_tp_rank=rank,
        expert_tp_size=size,
        global_intermediate_size=I,
        local_intermediate_size=local_i,
    )


def test_engine_rejects_unpartitioned_or_wrong_rank_tp_banks():
    model = SimpleNamespace(dsv4_args=SimpleNamespace(moe_inter_dim=I))
    cfg = SimpleNamespace(model_config=model, tp_info=DistributedInfo(1, 2))
    _validate_dsv4_expert_partition(cfg, _bundle(rank=1))
    with pytest.raises(RuntimeError, match="not rank-sharded"):
        _validate_dsv4_expert_partition(cfg, _bundle(rank=0))
    with pytest.raises(RuntimeError, match="not rank-sharded"):
        _validate_dsv4_expert_partition(
            cfg, ExpertBanks("ds_fp4", _empty_rank_banks(I))
        )


def test_legacy_unannotated_full_bank_is_allowed_only_for_tp1():
    model = SimpleNamespace(dsv4_args=SimpleNamespace(moe_inter_dim=I))
    cfg = SimpleNamespace(model_config=model, tp_info=DistributedInfo(0, 1))
    _validate_dsv4_expert_partition(cfg, ExpertBanks("ds_fp4", _empty_rank_banks(I)))


def test_tp_cpu_affinity_is_disjoint_and_leaves_explicit_tail(monkeypatch):
    monkeypatch.setattr(
        "freetoken.moe.cpu_executor.physical_core_cpus", lambda: list(range(32))
    )
    assert resolve_tp_threads_and_affinity(12, 0, 2) == (12, list(range(12)))
    assert resolve_tp_threads_and_affinity(12, 1, 2) == (12, list(range(12, 24)))
    assert resolve_tp_threads_and_affinity(0, 0, 2) == (16, list(range(16)))
    assert resolve_tp_threads_and_affinity(0, 1, 2) == (16, list(range(16, 32)))
    with pytest.raises(ValueError, match="cannot allocate"):
        resolve_tp_threads_and_affinity(17, 0, 2)


def test_tp_flag_coordinators_use_rank_indexed_tail_cores(monkeypatch):
    monkeypatch.setattr(
        "freetoken.moe.cpu_executor.physical_core_cpus", lambda: list(range(32))
    )
    rank0 = resolve_flag_coord_affinity(12, 12, list(range(12)), tp_rank=0, tp_size=2)
    rank1 = resolve_flag_coord_affinity(
        12, 12, list(range(12, 24)), tp_rank=1, tp_size=2
    )
    assert rank0 == (24, 12, list(range(12)))
    assert rank1 == (25, 12, list(range(12, 24)))

    # Auto sizing instead reserves the last core inside each rank's own pool.
    assert resolve_flag_coord_affinity(
        0, 16, list(range(16)), tp_rank=0, tp_size=2
    ) == (15, 15, list(range(15)))
    assert resolve_flag_coord_affinity(
        0, 16, list(range(16, 32)), tp_rank=1, tp_size=2
    ) == (31, 15, list(range(16, 31)))

    # A fully subscribed explicit pool has no dedicated tail and stays unpinned.
    assert resolve_flag_coord_affinity(
        16, 16, list(range(16)), tp_rank=0, tp_size=2
    ) == (-1, 16, list(range(16)))


def test_dsv4_shared_expert_is_added_after_routed_result_once():
    events = []

    class FakeGate(nn.Module):
        def forward(self, x, input_ids):
            events.append("gate")
            return torch.ones(x.shape[0], 1), torch.zeros(
                x.shape[0], 1, dtype=torch.int64
            )

    class FakeShared(nn.Module):
        def forward(self, x):
            events.append("shared")
            return torch.full_like(x, 3)

    class FakeRouted(nn.Module):
        def routed_forward(self, x, weights, indices):
            events.append("routed")
            assert weights.dtype == torch.float32
            assert indices.dtype == torch.int32
            # This value represents the already-all-reduced routed partial.
            return torch.full_like(x, 5)

    moe = MoE.__new__(MoE)
    nn.Module.__init__(moe)
    moe.dim = 4
    moe.gate = FakeGate()
    moe.shared_experts = FakeShared()
    moe.experts = FakeRouted()

    out = moe(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.int64))
    assert events == ["gate", "shared", "routed"]
    assert torch.equal(out, torch.full((1, 2, 4), 8.0))
