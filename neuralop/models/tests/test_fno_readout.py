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


def test_resolution_invariant_readout_non_unit_measure():
    """integral reduce with non-unit measure_per_dim should scale pre-head pooling by domain volume.

    The measure scales the spatially-pooled tensor before the linear head, so the
    relationship is head(6 * pooled) not 6 * head(pooled) — these differ by the bias
    term. We zero the bias to isolate the scaling behaviour.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    readout_unit = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[1.0, 1.0]
    ).to(device)
    readout_scaled = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[2.0, 3.0]
    ).to(device)

    # Share weights and zero bias so head is purely linear: head(6x) == 6 * head(x)
    readout_scaled.head.load_state_dict(readout_unit.head.state_dict())
    with torch.no_grad():
        readout_unit.head.bias.zero_()
        readout_scaled.head.bias.zero_()

    x = torch.randn(2, 4, 8, 8, device=device)
    with torch.no_grad():
        out_unit = readout_unit(x)
        out_scaled = readout_scaled(x)

    # domain volume = 2.0 * 3.0 = 6.0, so scaled output should be exactly 6x unit
    torch.testing.assert_close(out_scaled, out_unit * 6.0)


def test_resolution_invariant_readout_measure_length_mismatch():
    """measure_per_dim with wrong length should raise ValueError on forward."""
    readout = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[1.0, 1.0, 1.0]
    )
    x = torch.randn(2, 4, 8, 8)  # 2 spatial dims, but measure_per_dim has 3
    with pytest.raises(ValueError, match="measure_per_dim"):
        readout(x)


def test_resolution_invariant_readout_mean_vs_integral_differ():
    """mean and integral reduce should produce numerically different outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    readout_mean = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="mean", measure_per_dim=[2.0, 2.0]
    ).to(device)
    readout_integral = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[2.0, 2.0]
    ).to(device)

    # Share weights so only the reduce mode differs
    readout_integral.head.load_state_dict(readout_mean.head.state_dict())

    x = torch.randn(2, 4, 8, 8, device=device)
    with torch.no_grad():
        out_mean = readout_mean(x)
        out_integral = readout_integral(x)

    assert not torch.allclose(out_mean, out_integral), (
        "mean and integral outputs should differ when measure_per_dim != 1"
    )
