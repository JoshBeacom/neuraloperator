from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn


class ResolutionInvariantReadout(nn.Module):
    """Resolution-invariant readout for neural operator field outputs.

    Maps tensors of shape ``(B, C, *spatial)`` to ``(B, out_dim)`` by
    spatially pooling the field and applying a learned head.  The pooling
    mode is either a plain mean (``reduce="mean"``) or a physical integral
    approximation (``reduce="integral"``) that scales the mean by the domain
    volume so that the result is independent of grid resolution.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input field ``(B, C, *spatial)``.
    out_dim : int
        Number of output dimensions.
    reduce : {"mean", "integral"}, optional
        Spatial reduction mode.  ``"mean"`` computes the spatial mean.
        ``"integral"`` multiplies the mean by the domain volume so that
        ``mean(field) * measure_per_dim**n_dims == integral(field) * dvol``.
        Default is ``"mean"``.
    measure_per_dim : float or sequence of float, optional
        Physical domain measure per spatial dimension, used only when
        ``reduce="integral"``.  If a scalar, the domain volume is
        ``measure_per_dim ** n_spatial_dims``.  If a sequence, its length
        must match the number of spatial dimensions and the domain volume is
        the product of the entries (e.g., pass ``[Lx, Ly, Lz]`` for a
        non-cubic 3-D domain).  Defaults to ``1.0``.

        For non-cubic domains pass a sequence with one entry per dimension,
        e.g. ``measure_per_dim=[Lx, Ly, Lz]`` for a box of side-lengths
        *Lx*, *Ly*, *Lz* in Å (or any consistent length unit).
    head : {"linear", "mlp"}, optional
        Projection head applied after pooling.  ``"linear"`` uses a single
        ``nn.Linear``; ``"mlp"`` uses a two-layer MLP with an activation.
        Default is ``"linear"``.
    mlp_hidden_dim : int, optional
        Hidden width of the MLP head.  Defaults to ``in_channels`` when
        ``head="mlp"``.
    activation : nn.Module, optional
        Activation function for the MLP head.  Defaults to ``nn.GELU()``.
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

        # _measure_scalar is None for the sequence case, non-None for the scalar
        # case.  It acts as a sentinel in _measure_product to select the right
        # computation path (.prod() vs .pow(n_dims)).  _measure_buffer holds the
        # raw values as a registered tensor so they move with the module across
        # devices/dtypes and appear in the state dict — different shapes for scalar
        # (0-D) vs sequence (1-D) mean cross-loading between the two raises a
        # clear PyTorch size-mismatch error rather than silently giving wrong results.
        #
        # Copy caller-provided sequences so later external mutation cannot
        # invalidate the cached product or dimension checks.
        if isinstance(measure_per_dim, (float, int)):
            self.measure_per_dim = float(measure_per_dim)
            # Sentinel for the scalar path in _measure_product(): use pow(n_dims)
            # rather than per-dimension product.
            self._measure_scalar = self.measure_per_dim
            self.register_buffer(
                "_measure_buffer",
                torch.tensor(self.measure_per_dim, dtype=torch.float32),
            )
        else:
            values = list(measure_per_dim)
            self.measure_per_dim = values
            self.register_buffer(
                "_measure_buffer",
                torch.tensor(values, dtype=torch.float32),
            )
            self._measure_scalar = None

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

    def _measure_product(
        self, n_spatial_dims: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self._measure_scalar is None:
            # Sequence case: validate length, then multiply the per-dimension measures.
            if len(self.measure_per_dim) != n_spatial_dims:
                raise ValueError(
                    f"measure_per_dim has length {len(self.measure_per_dim)},"
                    f" expected {n_spatial_dims}."
                )
            return self._measure_buffer.to(device=device, dtype=dtype).prod()
        else:
            value = self._measure_buffer.to(device=device, dtype=dtype)
            return value.pow(n_spatial_dims)

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
