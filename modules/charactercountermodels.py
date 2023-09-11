import torch

import torch.nn as nn

from torchsummary import summary

from modules.pyramidpooling import TemporalPyramidPooling
# from pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_character_counter'
]


class CharacterCounterNet(nn.Module):
    def __init__(self, padding: str, pyramid_pooling=False, outputs:int=17):
        super().__init__()

        if not (padding == 'same' or padding == 'valid'):
            raise Exception('Padding has to be same or valid')

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=padding),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=padding),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=padding),
            nn.ReLU(),
        )
        
        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.head = nn.Sequential(
            nn.Linear(4096, outputs),
            nn.Softmax(1)
        )

        

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)

        return self.head(x)


@register_model
def PHOSCnet_character_counter(outputs=17, **kwargs):
    return CharacterCounterNet('same', outputs)

if __name__ == '__main__':
    model = CharacterCounterNet(padding='same', outputs=17).to('cuda')

    summary(model, (3, 50, 250))

    x = torch.randn((5, 3, 50, 250)).to('cuda')

    y = model(x)

    print(y)
    print(y.shape)