import torch

import torch.nn as nn

import sys
#sys.path.append("/home/aniket/PHOSCnetNor/modules/")
#sys.path.append("/home/aniket/PHOSCnetNor/")


from modules.pyramidpooling import SpatialPyramidPooling #spatial_pyramid_pool
from modules.pyramidpooling import TemporalPyramidPooling
# from pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model
import sys
#sys.path.append("/home/aniket/PHOSCnetNor/modules/")
#sys.path.append("/home/aniket/PHOSCnetNor/")



__all__ = [
    'PHOSCnet_temporalpooling_paper',
    "FixedPatchPrompter"
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

        self.temporal_pool = TemporalPyramidPooling([1, 2, 4])
        #self.spatial_pool = SpatialPyramidPooling([1,2,5])
        self.spp = 1
        
        if self.spp ==1:
            self.spatial_pool = SpatialPyramidPooling([1,2,4])


        
        if self.spp ==1:
            self.phos = nn.Sequential(
                    nn.Linear(10752, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(4096, 180),
                    nn.ReLU()
            )

            self.phoc = nn.Sequential(
                nn.Linear(10752, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(4096, 646),
                nn.Sigmoid()
            )

        else:

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
        #x = self.temporal_pool(x)

        #print("\n\t 1.x.shape:",x.shape)


        x= self.spatial_pool(x)
        
        #print("\n\t 2.x.shape spatial_pool:",x.shape)

        return {'phos': self.phos(x), 'phoc': self.phoc(x)}


@register_model
def PHOSCnet_temporalpooling_paper(**kwargs):
    return PHOSCnet()


class FixedPatchPrompterModel(nn.Module):
    def __init__(self):
        super(FixedPatchPrompterModel, self).__init__()
        #self.isize = args.image_size
        #self.psize = args.prompt_size

        self.h = 50
        self.w = 250

        self.patch = nn.Parameter(torch.randn([1, 3, self.h, self.w]))

    def forward(self, x):
        #prompt = torch.zeros((1, 3, self.h, self.w],require_grad = True))#.to("cuda:1")#cuda(":1")

        #prompt = torch.zeros(1, 3, self.h, self.w)

        #prompt[:, :, :self.h, :self.w] = self.patch

        #print("\n\t x.shape:",x.shape,"\t p",prompt.shape)
        #print("\n\t x.device:",x.device," prompt:",self.patch.device)
        return x + self.patch #prompt

@register_model
def FixedPatchPrompter(**kwargs):
    return FixedPatchPrompterModel()


if __name__ == '__main__':
    from torchsummary import summary

    model = PHOSCnet()

    summary(model.cuda(), (3, 50, 250))
