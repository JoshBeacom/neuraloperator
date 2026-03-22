import pytest
import torch

from neuralop.models import FNO, ResolutionInvariantReadout


@pytest.mark.parametrize("reduce", ["mean", "integral"])
def test_fno_resolution_invariant_readout_shapes(reduce):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dim = 7
    model = FNO(
        n_modes=(8, 8),
        in_channels=2,
        out_channels=4,
        hidden_channels=12,
        n_layers=2,
        readout=ResolutionInvariantReadout(
            in_channels=4,
            out_dim=out_dim,
            reduce=reduce,
            measure_per_dim=[1.0, 1.0],
        ),
    ).to(device)

    x_16 = torch.randn(3, 2, 16, 16, device=device)
    x_32 = torch.randn(3, 2, 32, 32, device=device)

    y_16 = model(x_16)
    y_32 = model(x_32)

    assert y_16.shape == (3, out_dim)
    assert y_32.shape == (3, out_dim)


def test_fno_resolution_invariant_readout_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FNO(
        n_modes=(8, 8),
        in_channels=2,
        out_channels=4,
        hidden_channels=12,
        n_layers=2,
        readout=ResolutionInvariantReadout(
            in_channels=4,
            out_dim=5,
            reduce="mean",
            head="mlp",
            mlp_hidden_dim=8,
        ),
    ).to(device)

    x = torch.randn(4, 2, 16, 16, device=device)
    target = torch.randn(4, 5, device=device)

    out = model(x)
    loss = (out - target).pow(2).mean()
    loss.backward()

    has_grad = any(param.grad is not None for param in model.parameters())
    assert has_grad
