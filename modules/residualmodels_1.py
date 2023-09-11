import torch

import torch.nn as nn
import torchsummary

from modules.pyramidpooling import TemporalPyramidPooling
# from pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'RPnet',
    'ResNet18Phosc',
    'ResNet18Phosc_preload_conv'
]


class ResBlockProjectionPaddingSame(nn.Module):
    def __init__(self, in_channels, out_channels, upsample, activation):
        super().__init__()
        if upsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding='same')
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, padding='same'),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding='same')
            self.shortcut = nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding='same')

        self.act1 = activation()
        self.act2 = activation()

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + shortcut

        return self.act2(x)


class ResidualPHOSCnet(nn.Module):
    def __init__(self, in_channels=3, activation=nn.ReLU) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64,
                      kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            activation(),

            ResBlockProjectionPaddingSame(64, 64, False, activation),

            nn.MaxPool2d((2, 2), stride=2),

            ResBlockProjectionPaddingSame(64, 128, True, activation),

            nn.MaxPool2d((2, 2), stride=2),

            ResBlockProjectionPaddingSame(128, 256, True, activation),
            ResBlockProjectionPaddingSame(256, 256, False, activation),
            ResBlockProjectionPaddingSame(256, 256, False, activation),

            ResBlockProjectionPaddingSame(256, 512, True, activation),
            ResBlockProjectionPaddingSame(512, 512, False, activation)
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


class ResBlockProjection(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, residual_tail=''):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)

        self.act1 = activation()
        self.act2 = activation()

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + shortcut

        return self.act2(x)


class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, activation, res_start_dim=64, phos_size=165, phoc_size=604):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, res_start_dim, kernel_size=7, stride=2),
            nn.BatchNorm2d(res_start_dim),
            activation(),

            nn.MaxPool2d(kernel_size=3, stride=2),

            # res blocks
            resblock(res_start_dim, res_start_dim, downsample=False,
                     activation=activation, residual_tail='init'),
            resblock(res_start_dim, res_start_dim,
                     downsample=False, activation=activation),
            

            resblock(res_start_dim, res_start_dim*2,
                     downsample=True, activation=activation),
            resblock(res_start_dim*2, res_start_dim*2,
                     downsample=False, activation=activation),
            

            resblock(res_start_dim*2, res_start_dim*4,
                     downsample=True, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            

            resblock(res_start_dim*4, res_start_dim*8,
                     downsample=True, activation=activation),
            resblock(res_start_dim*8, res_start_dim*8, downsample=False,
                     activation=activation, residual_tail='last')
        )

        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.phos = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, phos_size),
            nn.ReLU()
        )

        self.phoc = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, phoc_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)

        return {'phos': self.phos(x), 'phoc': self.phoc(x)}

    def preload_conv_layer(self, weights_file):
        self.conv.load_state_dict(torch.load(weights_file))

class ResNet34(nn.Module):
    def __init__(self, in_channels, resblock, activation, res_start_dim=64, outputs=200):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, res_start_dim, kernel_size=7, stride=2),
            nn.BatchNorm2d(res_start_dim),
            activation(),

            nn.MaxPool2d(kernel_size=3, stride=2),

            # res blocks
            resblock(res_start_dim, res_start_dim, downsample=False,
                     activation=activation, residual_tail='init'),
            resblock(res_start_dim, res_start_dim,
                     downsample=False, activation=activation),
            resblock(res_start_dim, res_start_dim,
                     downsample=False, activation=activation),
            resblock(res_start_dim, res_start_dim,
                     downsample=False, activation=activation),
            resblock(res_start_dim, res_start_dim,
                     downsample=False, activation=activation),
            resblock(res_start_dim, res_start_dim,
                     downsample=False, activation=activation),

            resblock(res_start_dim, res_start_dim*2,
                     downsample=True, activation=activation),
            resblock(res_start_dim*2, res_start_dim*2,
                     downsample=False, activation=activation),
            resblock(res_start_dim*2, res_start_dim*2,
                     downsample=False, activation=activation),
            resblock(res_start_dim*2, res_start_dim*2,
                     downsample=False, activation=activation),
            resblock(res_start_dim*2, res_start_dim*2,
                     downsample=False, activation=activation),
            resblock(res_start_dim*2, res_start_dim*2,
                     downsample=False, activation=activation),
            resblock(res_start_dim*2, res_start_dim*2,
                     downsample=False, activation=activation),
            resblock(res_start_dim*2, res_start_dim*2,
                     downsample=False, activation=activation),

            resblock(res_start_dim*2, res_start_dim*4,
                     downsample=True, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),
            resblock(res_start_dim*4, res_start_dim*4,
                     downsample=False, activation=activation),

            resblock(res_start_dim*4, res_start_dim*8,
                     downsample=True, activation=activation),
            resblock(res_start_dim*8, res_start_dim*8,
                     downsample=False, activation=activation),
            resblock(res_start_dim*8, res_start_dim*8,
                     downsample=False, activation=activation),
            resblock(res_start_dim*8, res_start_dim*8,
                     downsample=False, activation=activation),
            resblock(res_start_dim*8, res_start_dim*8,
                     downsample=False, activation=activation),
            resblock(res_start_dim*8, res_start_dim*8, downsample=False,
                     activation=activation, residual_tail='last')
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
def RPnet(**kwargs):
    return ResidualPHOSCnet()

@register_model
def ResNet18Phosc(**kwargs):
    return ResNet18(3, ResBlockProjection, nn.ReLU, phos_size=kwargs['phos_size'], phoc_size=kwargs['phoc_size'])

@register_model
def ResNet18Phosc_preload_conv(**kwargs):
    model = ResNet18(3, ResBlockProjection, nn.ReLU, phos_size=kwargs['phos_size'], phoc_size=kwargs['phoc_size'])
    model.preload_conv_layer('logs_weights/ResNet18Phosc/conv_layers.pt')

    # freeze convolutional part
    # for param in model.conv.parameters():
    #     param.requires_grad = False

    return model



if __name__ == '__main__':
    from torchsummary import summary

    # model = ResNet18(3, ResBlockProjection, nn.ReLU, phos_size=kwargs['phos_size'], phoc_size=kwargs['phoc_size'])
    model = ResNet18Phosc(phos_size=165, phoc_size=604).to('cuda')

    #model = ResNet18Phosc_preload_conv(phos_size=165, phoc_size=604)
    # model = RPnet(phos_size=165, phoc_size=604)

    summary(model, (3, 50, 250))

    # model.load_state_dict(torch.load('logs_weights/ResNet18Phosc/epoch41.pt'))

    # torch.save(model.conv.state_dict(), 'logs_weights/ResNet18Phosc/conv_layers.pt')

    # model.preload_conv_layer('logs_weights/ResNet18Phosc/conv_layers.pt')


    x = torch.randn((5, 3, 50, 250)).to('cuda')

    vector_dict = model(x)

    vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)

    print(vectors.shape)
