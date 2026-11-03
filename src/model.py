from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn


class PurePyTorchS6Block(nn.Module):
    def __init__(self, d_model: int = 512, d_state: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        state_values = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
            d_model, 1
        )
        self.A_log = nn.Parameter(torch.log(state_values))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = x.shape
        original_dtype = x.dtype

        with torch.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            A = -torch.exp(self.A_log.float())
            projected = self.x_proj(x_float)
            delta, B_proj, C_proj = torch.split(
                projected, [1, self.d_state, self.d_state], dim=-1
            )
            delta = F.softplus(self.dt_proj(delta))
            deltaA = torch.clamp(
                torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_model, self.d_state)),
                max=0.999,
            )
            deltaB = delta.unsqueeze(-1) * B_proj.unsqueeze(2)

            state = torch.zeros(
                batch_size,
                d_model,
                self.d_state,
                device=x.device,
                dtype=torch.float32,
            )
            outputs = []
            for step in range(sequence_length):
                state = (
                    deltaA[:, step] * state
                    + deltaB[:, step] * x_float[:, step, :, None]
                )
                outputs.append((state * C_proj[:, step, None, :]).sum(dim=-1))

            output = self.out_proj(torch.stack(outputs, dim=1))

        return output.to(original_dtype)


class StreamlinedFFA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        srm_kernel = (
            torch.tensor(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=torch.float32,
            )
            / 12.0
        )
        self.srm = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False, groups=3)
        self.srm.weight.data = srm_kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        self.srm.weight.requires_grad = False

        self.bayar = nn.Conv2d(3, 4, kernel_size=5, padding=2, bias=False)

        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        )
        self.laplacian = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.laplacian.weight.data = laplacian_kernel.view(1, 1, 3, 3).repeat(
            3, 1, 1, 1
        )
        self.laplacian.weight.requires_grad = False

        self.fuse = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve the notebook's fixed Bayar-filter normalization at runtime.
        with torch.no_grad():
            weights = self.bayar.weight.detach().clone()
            weights[:, :, 2, 2] = 0
            weights = weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-7)
            weights[:, :, 2, 2] = -1.0
            self.bayar.weight.copy_(weights)

        srm_features = self.srm(x)
        bayar_features = self.bayar(x)
        laplacian_features = self.laplacian(x)
        return self.fuse(
            torch.cat([srm_features, bayar_features, laplacian_features], dim=1)
        )


class DevelopedF2Mamba(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            "resnet34", pretrained=False, features_only=True
        )
        self.s6_bottleneck = PurePyTorchS6Block(d_model=512, d_state=16)
        self.ffa = StreamlinedFFA()

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.gate_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.out_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        c0, c1, c2, c3, c4 = features

        batch_size, channels, height, width = c4.shape
        c4_flat = c4.flatten(2).permute(0, 2, 1)
        s6_output = self.s6_bottleneck(c4_flat)
        s6_map = s6_output.permute(0, 2, 1).view(batch_size, channels, height, width)

        d4 = self.conv4(torch.cat([self.up4(s6_map), c3], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d4), c2], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), c1], dim=1))
        d1 = self.conv1(torch.cat([self.up1(d2), c0], dim=1))
        d0 = self.conv0(self.up0(d1))

        forensic_features = self.ffa(x)
        gate = torch.sigmoid(self.gate_conv(d0))
        global_prediction = torch.sigmoid(self.out_head(d0))
        forensic_prediction = torch.sigmoid(forensic_features)
        return gate * global_prediction + (1.0 - gate) * forensic_prediction
