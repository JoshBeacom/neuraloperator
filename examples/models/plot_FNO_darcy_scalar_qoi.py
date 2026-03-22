"""
FNO scalar QoI prediction with resolution-invariant readout
===========================================================

This example trains an FNO to predict a scalar quantity of interest (QoI)
from Darcy-Flow fields. We define the target as:

``y_scalar = mean(y_field)``

and use a resolution-invariant readout to map ``(B, C, H, W) -> (B, 1)``.
The model is trained on one resolution and evaluated zero-shot at multiple
resolutions.
"""

import torch
import torch.nn as nn

from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.models import FNO, ResolutionInvariantReadout
from neuralop.training import AdamW


device = "cpu"


class ScalarQOILoss(nn.Module):
    """MSE against scalar QoI computed from y, ignoring extra kwargs."""

    def forward(self, out, y, **kwargs):
        y_scalar = y.reshape(y.shape[0], -1).mean(dim=1, keepdim=True)
        return torch.mean((out - y_scalar) ** 2)


train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=256,
    batch_size=32,
    n_tests=[64, 64],
    test_resolutions=[16, 32],
    test_batch_sizes=[32, 32],
)
data_processor = data_processor.to(device)

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
).to(device)

optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

qoi_loss = ScalarQOILoss()

trainer = Trainer(
    model=model,
    n_epochs=3,
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
    for resolution in [16, 32]:
        sample = next(iter(test_loaders[resolution]))
        sample = data_processor.preprocess(sample)
        out = model(**sample)
        y_scalar = (
            sample["y"].reshape(sample["y"].shape[0], -1).mean(dim=1, keepdim=True)
        )
        mse = torch.mean((out - y_scalar) ** 2).item()
        print(f"Resolution {resolution}x{resolution} scalar QoI MSE: {mse:.6f}")
