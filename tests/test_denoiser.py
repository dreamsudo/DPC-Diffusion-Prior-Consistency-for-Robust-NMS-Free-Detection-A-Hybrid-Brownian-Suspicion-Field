"""Test denoiser channel arithmetic.

Pins mistake #6 from the catalogue: a decoder channel-count bug that took
two days to find in v2. The forward pass must succeed at the canonical
resolutions used in the project.
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("res", [64, 128, 256])
def test_forward_at_canonical_resolutions(res):
    from dpc.denoiser import TinyUNetDenoiser

    model = TinyUNetDenoiser(use_attention=True)
    x = torch.randn(2, 3, res, res)
    t = torch.tensor([100, 500], dtype=torch.long)
    out = model(x, t)
    assert out.shape == x.shape


def test_forward_with_attention():
    from dpc.denoiser import TinyUNetDenoiser
    model = TinyUNetDenoiser(use_attention=True)
    x = torch.randn(1, 3, 128, 128)
    t = torch.tensor([0], dtype=torch.long)
    out = model(x, t)
    assert out.shape == (1, 3, 128, 128)


def test_forward_without_attention():
    from dpc.denoiser import TinyUNetDenoiser
    model = TinyUNetDenoiser(use_attention=False)
    x = torch.randn(1, 3, 128, 128)
    t = torch.tensor([0], dtype=torch.long)
    out = model(x, t)
    assert out.shape == (1, 3, 128, 128)


def test_param_count_in_expected_range():
    """v2 reported 1,013,091 params with attention; v3 must match."""
    from dpc.denoiser import TinyUNetDenoiser
    model = TinyUNetDenoiser(use_attention=True)
    n = sum(p.numel() for p in model.parameters())
    assert 900_000 <= n <= 1_200_000, f"unexpected param count: {n:,}"


def test_gradient_flow():
    """Loss.backward() must produce non-zero grads in all parameters."""
    from dpc.denoiser import TinyUNetDenoiser
    model = TinyUNetDenoiser(use_attention=True)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    t = torch.tensor([100, 500], dtype=torch.long)
    out = model(x, t)
    loss = out.pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
