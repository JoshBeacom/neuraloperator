"""
FNO scalar QoI prediction with resolution-invariant readout
===========================================================

This example trains an FNO to predict a scalar quantity of interest (QoI)
from Darcy-Flow fields. We define the target as:

``y_scalar = mean(y_field)``

and use a resolution-invariant readout to map ``(B, C, H, W) -> (B, 1)``.
The model is trained on one resolution and evaluated zero-shot at multiple
resolutions. For the real Darcy dataset path, we disable output-field
normalization because the target is a derived scalar QoI rather than the full
output field.

Pass ``--synthetic-data`` to run a fast smoke test without downloading the
Darcy dataset from Zenodo.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.models import FNO, ResolutionInvariantReadout
from neuralop.training import AdamW


class ScalarQOILoss(nn.Module):
    """MSE against scalar QoI computed from y, ignoring extra kwargs."""

    def forward(self, out, y, **kwargs):
        y_scalar = y.reshape(y.shape[0], -1).mean(dim=1, keepdim=True)
        return torch.mean((out - y_scalar) ** 2)


class SyntheticDarcyLikeDataset(Dataset):
    """Small deterministic dataset for smoke testing the QoI example."""

    def __init__(self, n_samples: int, resolution: int):
        self.n_samples = n_samples
        coords = torch.linspace(0.0, 1.0, resolution)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.base_x = (xx + yy).unsqueeze(0)
        self.base_y = (torch.sin(torch.pi * xx) * torch.cos(torch.pi * yy)).unsqueeze(0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        scale = (idx + 1) / self.n_samples
        x = self.base_x * scale
        y = self.base_y * (1.0 + 0.5 * scale)
        return {"x": x, "y": y}


def build_synthetic_loaders(batch_size: int):
    """Return tiny in-memory loaders for fast example validation."""
    train_loader = DataLoader(
        SyntheticDarcyLikeDataset(n_samples=32, resolution=16),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loaders = {
        16: DataLoader(
            SyntheticDarcyLikeDataset(n_samples=8, resolution=16), batch_size=batch_size
        ),
        32: DataLoader(
            SyntheticDarcyLikeDataset(n_samples=8, resolution=32), batch_size=batch_size
        ),
    }
    return train_loader, test_loaders, DefaultDataProcessor()


def build_model(device: str):
    model = FNO(
        n_modes=(8, 8),
        in_channels=1,
        out_channels=16,
        hidden_channels=24,
        projection_channel_ratio=2,
        readout=ResolutionInvariantReadout(
            in_channels=16,
            out_dim=1,
            reduce="mean",
            head="mlp",
            mlp_hidden_dim=32,
        ),
    )
    return model.to(device)


def load_data(args):
    if args.synthetic_data:
        return build_synthetic_loaders(batch_size=args.batch_size)

    # Disable output normalization here because this example learns a scalar
    # QoI derived from the field, not the field itself.
    return load_darcy_flow_small(
        n_train=256,
        batch_size=args.batch_size,
        n_tests=[64, 64],
        test_resolutions=[16, 32],
        test_batch_sizes=[args.batch_size, args.batch_size],
        encode_output=False,
    )


def run_example(args):
    device = args.device
    train_loader, test_loaders, data_processor = load_data(args)
    data_processor = data_processor.to(device)
    model = build_model(device)

    optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    qoi_loss = ScalarQOILoss()

    trainer = Trainer(
        model=model,
        n_epochs=args.epochs,
        device=device,
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=1,
        use_distributed=False,
        verbose=True,
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=qoi_loss,
        eval_losses={"qoi_mse": qoi_loss},
    )

    model.eval()
    with torch.no_grad():
        for resolution in sorted(test_loaders):
            sample = next(iter(test_loaders[resolution]))
            sample = data_processor.preprocess(sample)
            out = model(sample["x"])
            y_scalar = (
                sample["y"].reshape(sample["y"].shape[0], -1).mean(dim=1, keepdim=True)
            )
            mse = torch.mean((out - y_scalar) ** 2).item()
            print(f"Resolution {resolution}x{resolution} scalar QoI MSE: {mse:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an FNO with a resolution-invariant readout for scalar QoI prediction."
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use a tiny in-memory dataset instead of downloading Darcy-Flow data.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for train and test loaders.",
    )
    return parser.parse_args()


def main():
    run_example(parse_args())


if __name__ == "__main__":
    main()
