from __future__ import annotations

from unittest.mock import patch

import pytest

from freetoken.server.args import parse_args


class _Config:
    def to_dict(self):
        return {
            "architectures": ["DeepseekV4ForCausalLM"],
            "torch_dtype": "bfloat16",
        }


def _parse(*extra: str):
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        args, run_shell = parse_args(["--model", "/models/dsv4", *extra])
    assert run_shell is False
    return args


def test_parse_swa_full_tokens_ratio():
    args = _parse("--swa-full-tokens-ratio", "0.025")
    assert args.swa_full_tokens_ratio == pytest.approx(0.025)
    assert args.swa_num_pages_override is None


def test_parse_swa_num_pages():
    args = _parse("--swa-num-pages", "51")
    assert args.swa_num_pages_override == 51


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "nan", "not-a-number"])
def test_reject_invalid_swa_full_tokens_ratio(value: str):
    with pytest.raises(SystemExit):
        _parse("--swa-full-tokens-ratio", value)


def test_reject_ratio_and_absolute_swa_capacity_together():
    with pytest.raises(SystemExit):
        _parse(
            "--swa-full-tokens-ratio",
            "0.025",
            "--swa-num-pages",
            "51",
        )


def test_parse_default_reasoning_effort():
    assert _parse().default_reasoning_effort is None
    assert (
        _parse("--default-reasoning-effort", "high").default_reasoning_effort
        == "high"
    )


def test_reject_unknown_default_reasoning_effort():
    with pytest.raises(SystemExit):
        _parse("--default-reasoning-effort", "ultra")
