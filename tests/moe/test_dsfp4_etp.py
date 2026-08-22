"""DSV4 FP4 decode kernel parity for expert-intermediate TP."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dsfp4_full_expert_matches_sum_of_two_intermediate_partitions():
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    E, H, I, top_k = 8, 1024, 512, 4
    gen = torch.Generator(device="cpu").manual_seed(17)

    def u8(*shape, low=0, high=256):
        return torch.randint(
            low, high, shape, dtype=torch.uint8, generator=gen, device="cpu"
        ).cuda()

    gate_up = u8(E, 2 * I, H // 2)
    gate_up_scale = u8(E, 2 * I, H // 32, low=119, high=125)
    down = u8(E, H, I // 2)
    down_scale = u8(E, H, I // 32, low=119, high=125)
    x = (torch.randn(2, H, dtype=torch.bfloat16, generator=gen) * 0.1).cuda()
    ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(2)])
    ids = ids.to(device="cuda", dtype=torch.int32).contiguous()
    weights = torch.rand(2, top_k, generator=gen)
    weights = (weights / weights.sum(-1, keepdim=True)).cuda().float().contiguous()

    full = routed_experts_fp4(
        x, ids.clone(), weights, gate_up, gate_up_scale, down, down_scale, 7.0
    )
    local_i = I // 2
    partials = []
    for rank in (0, 1):
        lo, hi = rank * local_i, (rank + 1) * local_i
        local_gate_up = torch.cat(
            (gate_up[:, lo:hi], gate_up[:, I + lo : I + hi]), dim=1
        ).contiguous()
        local_gate_up_scale = torch.cat(
            (gate_up_scale[:, lo:hi], gate_up_scale[:, I + lo : I + hi]), dim=1
        ).contiguous()
        partials.append(
            routed_experts_fp4(
                x,
                ids.clone(),
                weights,
                local_gate_up,
                local_gate_up_scale,
                down[..., lo // 2 : hi // 2].contiguous(),
                down_scale[..., lo // 32 : hi // 32].contiguous(),
                7.0,
            )
        )

    etp = (partials[0] + partials[1]).to(torch.bfloat16)
    torch.cuda.synchronize()
    diff = (etp.float() - full.float()).abs()
    rel_rmse = diff.pow(2).mean().sqrt() / full.float().pow(2).mean().sqrt()
    # Partitioning changes the fp32 accumulation tree and rounds each rank's
    # hidden-size partial to bf16 before NCCL SUM.  The target geometry measured
    # ~0.37% relative RMSE; keep a conservative 1% regression ceiling.
    assert torch.isfinite(etp).all()
    assert rel_rmse.item() < 1e-2, rel_rmse.item()
