from __future__ import annotations

import pytest
import torch
from freetoken.engine.engine import _make_dummy_weight_state_dict


def test_dummy_weight_supports_e8m0_scales():
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is None:
        pytest.skip("PyTorch build does not expose float8_e8m0fnu")

    model_state = {
        "expert.weight_scale_inv": torch.empty((2, 3), dtype=e8m0),
        "trunk.weight": torch.empty((2, 3), dtype=torch.float8_e4m3fn),
    }
    state = _make_dummy_weight_state_dict(model_state, device=torch.device("cpu"))

    assert state["expert.weight_scale_inv"].dtype == e8m0
    assert torch.all(state["expert.weight_scale_inv"].view(torch.uint8) == 127)
    assert state["trunk.weight"].dtype == torch.float8_e4m3fn
    assert torch.all(state["trunk.weight"].view(torch.uint8) < 16)
