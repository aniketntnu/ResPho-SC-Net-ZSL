import torch

import torch.nn as nn
import torchsummary
import sys

sys.path.append("/global/D1/projects/ZeroShot_Word_Recognition/E2E/PHOSCnetNor")
from modules.pyramidpooling import TemporalPyramidPooling
# from pyramidpooling import TemporalPyramidPooling
###from modules.pyramidpooling import TemporalPyramidPooling
import torchvision.models as modelsRes

from timm.models.registry import register_model

__all__ = [
    'RPnet',
    "Resnet18Pretrain_biasAddition",
    "Resnet18Pretrain_biasAddition_IAM",
    'ResNet18Phosc_j',
    'ResNet18Phosc_preload_conv',
    "Resnet18Pretrain",
    "AttentionResNet"
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


class ResNet18_j(nn.Module):
    def __init__(self, in_channels, resblock, activation = nn.ReLU,res_start_dim=64, phos_size=180, phoc_size=646):
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


@register_model
def RPnet(**kwargs):
    return ResidualPHOSCnet()

@register_model
def ResNet18Phosc_j(**kwargs):
    return ResNet18_j(3, ResBlockProjection, nn.ReLU, phos_size=180, phoc_size=646)

@register_model
def ResNet18Phosc_preload_conv(**kwargs):
    model = ResNet18(3, ResBlockProjection, nn.ReLU, phos_size=kwargs['phos_size'], phoc_size=kwargs['phoc_size'])
    model.preload_conv_layer('logs_weights/ResNet18Phosc/conv_layers.pt')

    # freeze convolutional part
    # for param in model.conv.parameters():
    #     param.requires_grad = False

    return model


class ResNet18Pretrain(nn.Module):
    def __init__(self, n_out=0, in_channels=1, gpp_type='tpp', pooling_levels=3, pool_type='max_pool'):
        super().__init__()
        self.resnet18 = modelsRes.resnet18(pretrained=True)
        #resnet18 = modelsRes.resnet18(pretrained=True)

        #self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
        #self.fc = nn.Linear(512* 2* 8, n_out)
        
        """
        self.resnet18.conv1.padding = (0,0) # from (3,3) it is changed to (0,0)
        self.resnet18.conv1.bias = nn.Parameter(torch.zeros(64).cuda()) # initially bias was False Now It is added
        self.resnet18.maxpool.padding = (0,0) # from (1,1) it is changed to (0,0)

        self.resnet18.layer1[0].conv1.bias = nn.Parameter(torch.zeros(64).cuda()) # initially bias was False Now It is added
        self.resnet18.layer1[0].conv2.bias = nn.Parameter(torch.zeros(64).cuda()) # initially bias was False Now It is added

        self.resnet18.layer1[1].conv1.bias = nn.Parameter(torch.zeros(64).cuda()) # initially bias was False Now It is added
        self.resnet18.layer1[1].conv2.bias = nn.Parameter(torch.zeros(64).cuda()) # initially bias was False Now It is added

        self.resnet18.layer2[0].conv1.bias = nn.Parameter(torch.zeros(128).cuda()) # initially bias was False Now It is added
        self.resnet18.layer2[0].conv2.bias = nn.Parameter(torch.zeros(128).cuda()) # initially bias was False Now It is added

        self.resnet18.layer2[0].downsample[0].bias = nn.Parameter(torch.zeros(128).cuda()) # initially bias was False Now It is added

        self.resnet18.layer2[1].conv1.bias = nn.Parameter(torch.zeros(128).cuda()) # initially bias was False Now It is added
        self.resnet18.layer2[1].conv2.bias = nn.Parameter(torch.zeros(128).cuda()) # initially bias was False Now It is added

        self.resnet18.layer3[0].conv1.bias = nn.Parameter(torch.zeros(256).cuda()) # initially bias was False Now It is added
        self.resnet18.layer3[0].conv2.bias = nn.Parameter(torch.zeros(256).cuda()) # initially bias was False Now It is added

        self.resnet18.layer3[0].downsample[0].bias = nn.Parameter(torch.zeros(256).cuda()) # initially bias was False Now It is added
        self.resnet18.layer3[1].conv1.bias = nn.Parameter(torch.zeros(256).cuda()) # initially bias was False Now It is added
        self.resnet18.layer3[1].conv2.bias =nn.Parameter(torch.zeros(256).cuda()) # initially bias was False Now It is added

        self.resnet18.layer4[0].conv1.bias = nn.Parameter(torch.zeros(512).cuda()) # initially bias was False Now It is added
        self.resnet18.layer4[0].conv2.bias = nn.Parameter(torch.zeros(512).cuda()) # initially bias was False Now It is added

        self.resnet18.layer4[0].downsample[0].bias = nn.Parameter(torch.zeros(512).cuda()) # initially bias was False Now It is added
        self.resnet18.layer4[1].conv1.bias = nn.Parameter(torch.zeros(512).cuda()) # initially bias was False Now It is added
        self.resnet18.layer4[1].conv2.bias =nn.Parameter(torch.zeros(512).cuda()) # initially bias was False Now It is added
        """

        self.resnet18.conv1.padding = (0,0) # from (3,3) it is changed to (0,0)
        self.resnet18.conv1.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added
        self.resnet18.maxpool.padding = (0,0) # from (1,1) it is changed to (0,0)

        self.resnet18.layer1[0].conv1.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added
        self.resnet18.layer1[0].conv2.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added

        self.resnet18.layer1[1].conv1.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added
        self.resnet18.layer1[1].conv2.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added

        self.resnet18.layer2[0].conv1.bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added
        self.resnet18.layer2[0].conv2.bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added

        self.resnet18.layer2[0].downsample[0].bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added

        self.resnet18.layer2[1].conv1.bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added
        self.resnet18.layer2[1].conv2.bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added

        self.resnet18.layer3[0].conv1.bias = nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added
        self.resnet18.layer3[0].conv2.bias = nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added

        self.resnet18.layer3[0].downsample[0].bias = nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added
        self.resnet18.layer3[1].conv1.bias = nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added
        self.resnet18.layer3[1].conv2.bias =nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added

        self.resnet18.layer4[0].conv1.bias = nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added
        self.resnet18.layer4[0].conv2.bias = nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added

        self.resnet18.layer4[0].downsample[0].bias = nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added
        self.resnet18.layer4[1].conv1.bias = nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added
        self.resnet18.layer4[1].conv2.bias =nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added


        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])

        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])
        
        #""" for resnet 18,34
        self.phos1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 180),
            nn.ReLU()
        )

        self.phoc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 646),
            nn.Sigmoid()

        )
        

    def forward(self, input_tensor):

        y = input_tensor

        if y.shape[1] == 1:
            y = y.expand((y.shape[0], 3, *y.shape[2:]))
        x = self.resnet18(y)
        
        #print("\n\t x1:",x.shape)
        x = self.temporal_pool(x)
        
        #print("\n\t x2:",x.shape)
        return {'phos': self.phos1(x), 'phoc': self.phoc1(x),"x":x}

@register_model
def Resnet18Pretrain_biasAddition(**kwargs):
    return ResNet18Pretrain()

class ResNet18Pretrain_IAM(nn.Module):
    def __init__(self, n_out=0, in_channels=1, gpp_type='tpp', pooling_levels=3, pool_type='max_pool'):
        super().__init__()
        self.resnet18 = modelsRes.resnet18(pretrained=True)
        #resnet18 = modelsRes.resnet18(pretrained=True)

        #self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
        #self.fc = nn.Linear(512* 2* 8, n_out)
        
        """
        """

        self.resnet18.conv1.padding = (0,0) # from (3,3) it is changed to (0,0)
        self.resnet18.conv1.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added
        self.resnet18.maxpool.padding = (0,0) # from (1,1) it is changed to (0,0)

        self.resnet18.layer1[0].conv1.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added
        self.resnet18.layer1[0].conv2.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added

        self.resnet18.layer1[1].conv1.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added
        self.resnet18.layer1[1].conv2.bias = nn.Parameter(torch.zeros(64)) # initially bias was False Now It is added

        self.resnet18.layer2[0].conv1.bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added
        self.resnet18.layer2[0].conv2.bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added

        self.resnet18.layer2[0].downsample[0].bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added

        self.resnet18.layer2[1].conv1.bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added
        self.resnet18.layer2[1].conv2.bias = nn.Parameter(torch.zeros(128)) # initially bias was False Now It is added

        self.resnet18.layer3[0].conv1.bias = nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added
        self.resnet18.layer3[0].conv2.bias = nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added

        self.resnet18.layer3[0].downsample[0].bias = nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added
        self.resnet18.layer3[1].conv1.bias = nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added
        self.resnet18.layer3[1].conv2.bias =nn.Parameter(torch.zeros(256)) # initially bias was False Now It is added

        self.resnet18.layer4[0].conv1.bias = nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added
        self.resnet18.layer4[0].conv2.bias = nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added

        self.resnet18.layer4[0].downsample[0].bias = nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added
        self.resnet18.layer4[1].conv1.bias = nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added
        self.resnet18.layer4[1].conv2.bias =nn.Parameter(torch.zeros(512)) # initially bias was False Now It is added


        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])

        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])
        
        self.phos1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 165),
            nn.ReLU()
        )

        self.phoc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 604),
            nn.Sigmoid()

        )
        
    
    def forward(self, input_tensor):

        y = input_tensor

        if y.shape[1] == 1:
            y = y.expand((y.shape[0], 3, *y.shape[2:]))
        x = self.resnet18(y)
        
        #print("\n\t x1:",x.shape)
        x = self.temporal_pool(x)
        
        #print("\n\t x2:",x.shape)
        return {'phos': self.phos1(x), 'phoc': self.phoc1(x),"x":x}

@register_model
def Resnet18Pretrain_biasAddition_IAM(**kwargs):
    return ResNet18Pretrain_IAM()

class ResNet18PretrainAttention(nn.Module):
    def __init__(self, n_out=0, in_channels=1, gpp_type='tpp', pooling_levels=3, pool_type='max_pool'):
        super().__init__()
        resnet18 = modelsRes.resnet18(pretrained=True)
        #resnet18 = modelsRes.resnet18(pretrained=True)

        self.resnet18= nn.Sequential(*list(resnet18.children())[:-2])
        #self.fc = nn.Linear(512* 2* 8, n_out)
        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        #""" for resnet 18,34
        self.phos1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 180),
            nn.ReLU()
        )

        self.phoc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 646),
            nn.Sigmoid()

        )


    def forward(self, input_tensor):

        y = input_tensor

        if y.shape[1] == 1:
            y = y.expand((y.shape[0], 3, *y.shape[2:]))
        x = self.resnet18(y)

        print("\n\t x1:",x.shape)
        x = self.temporal_pool(x)

        #print("\n\t x2:",x.shape)
        return {'phos': self.phos1(x), 'phoc': self.phoc1(x),"x":x}


@register_model
def AttentionResnet(**kwargs):
    return ResNet18PretrainAttention()



if __name__ == '__main__':
    from torchsummary import summary

    #model = ResNet18Phosc_preload_conv(phos_size=165, phoc_size=604)
    #model = ResNet18PretrainAttention()


    model = ResNet18Pretrain()

    summary(model, (3, 50, 250))
    
    """
    model.load_state_dict(torch.load('logs_weights/ResNet18Phosc/epoch41.pt'))

    torch.save(model.conv.state_dict(), 'logs_weights/ResNet18Phosc/conv_layers.pt')

    model.preload_conv_layer('logs_weights/ResNet18Phosc/conv_layers.pt')


    x = torch.randn((5, 3, 50, 250)).to('cuda')

    vector_dict = model(x)

    vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)
    
    print(vectors.shape)

    """
