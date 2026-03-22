from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn


class ResolutionInvariantReadout(nn.Module):
    """Resolution-invariant readout for neural operator field outputs.

    Maps tensors of shape ``(B, C, *spatial)`` to ``(B, out_dim)`` by:
    1) reducing over spatial dimensions with either ``mean`` or ``integral``
    2) applying a head (linear or MLP)
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        reduce: Literal["mean", "integral"] = "mean",
        measure_per_dim: Optional[Union[float, Sequence[float]]] = None,
        head: Literal["linear", "mlp"] = "linear",
        mlp_hidden_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        if reduce not in {"mean", "integral"}:
            raise ValueError(f"reduce must be 'mean' or 'integral', got {reduce}.")
        if head not in {"linear", "mlp"}:
            raise ValueError(f"head must be 'linear' or 'mlp', got {head}.")

        self.reduce = reduce

        if measure_per_dim is None:
            measure_per_dim = 1.0
        self.measure_per_dim = measure_per_dim

        if head == "linear":
            self.head = nn.Linear(in_channels, out_dim)
        else:
            if mlp_hidden_dim is None:
                mlp_hidden_dim = in_channels
            if activation is None:
                activation = nn.GELU()
            self.head = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_dim),
                activation,
                nn.Linear(mlp_hidden_dim, out_dim),
            )

    def _measure_product(self, n_spatial_dims: int, device, dtype) -> torch.Tensor:
        if isinstance(self.measure_per_dim, (float, int)):
            measure = [float(self.measure_per_dim)] * n_spatial_dims
        else:
            measure = list(self.measure_per_dim)
            if len(measure) != n_spatial_dims:
                raise ValueError(
                    f"measure_per_dim has length {len(measure)}, expected {n_spatial_dims}."
                )

        return torch.tensor(measure, dtype=dtype, device=device).prod()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError(
                f"Expected input with shape (B, C, *spatial), got {tuple(x.shape)}"
            )

        spatial_dims = tuple(range(2, x.ndim))
        reduced = x.mean(dim=spatial_dims)

        if self.reduce == "integral":
            reduced = reduced * self._measure_product(
                n_spatial_dims=len(spatial_dims), device=x.device, dtype=x.dtype
            )

        return self.head(reduced)
