import torch

import torch.nn as nn

from modules.pyramidpooling import TemporalPyramidPooling
# from pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_temporalpooling',
]


class PHOSCnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding='same'),
            nn.ReLU(),
        )

        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.phos = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 165),
            nn.ReLU()
        )

        self.phoc = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 604),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)

        return {'phos': self.phos(x), 'phoc': self.phoc(x)}


@register_model
def PHOSCnet_temporalpooling(**kwargs):
    return PHOSCnet()


if __name__ == '__main__':
    from torchsummary import summary

    model = PHOSCnet()

    summary(model, (3, 50, 250))
