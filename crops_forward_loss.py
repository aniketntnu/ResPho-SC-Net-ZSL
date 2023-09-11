import os

import cv2 as cv
import torch
import torch.nn.functional as F
from skimage import io
from torch.optim import Adam
from torchvision.transforms import transforms

from modules.models import PHOSCnet_temporalpooling
from utils import generate_phoc_vector, generate_phos_vector

# creates phos and phoc vector for 'a'
batch_y_s = torch.tensor(generate_phos_vector('a'))
batch_y_c = torch.tensor(generate_phoc_vector('a'))

# expands to 12 phos and phoc vectors of 'a' and switches type to torch.float32
batch_y_s= batch_y_s.expand(12, -1).float()
batch_y_c= batch_y_c.expand(12, -1) # removing .float() will yield the following error in CE loss: RuntimeError: Found dtype Long but expected Float
"""
When creating the the phoc tensor from the generated phoc numpy array of generate_phoc_vector() the dtype is torch.int64 (torch.long).
This is possibly were the issue occurs. Datatypes needs to match in the loss functions.

The same dtype error will also occur with phos vector however this problem occurs in the backward pass.

Make sure every of all tensors are torch.float32 by retyping with .float()
"""

print('target vectors shape')
print(batch_y_s.dtype)
print(batch_y_c.dtype)

# print(batch_y_s.shape)
# print(batch_y_c.shape)

# creates the batch tensor for all images
batch_toTensor = torch.ones((len(os.listdir(os.path.join('image_data', 'crops', 'a'))), 3, 50, 250))
batch_manual = torch.ones((len(os.listdir(os.path.join('image_data', 'crops', 'a'))), 3, 50, 250))
print('batch shape')
print(batch_toTensor.shape)

# counter
c = 0

path_to_dir = os.path.join(os.getcwd(), 'image_data', 'crops', 'a') # the dir with crops of the letter a, here: image_data/crops/a

# preprocessing of crops
for file in os.listdir(path_to_dir):
    
    # read image
    image = cv.imread(os.path.join(path_to_dir, file))

    print('image before resizing')
    print(image.shape)

    # resize crops to H=50 W=250
    image = cv.resize(image, (250, 50))
    
    print('image after resizing')
    print(image.shape)

    # create transform object 
    transform = transforms.ToTensor()

    # transform images to tensors, normalizes and set dtype to torch.float32 https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    image_toTensor = transform(image)
    
    # transformation without toTensor, does the above statement but shows all steps
    image_manual = (torch.tensor(image) / 255).view(3, 50, 250).float()

    print(image_toTensor.shape)
    print(image_manual.shape)

    # adds image to batch
    batch_toTensor[c] = image_toTensor
    batch_manual[c] = image_manual
    c += 1


print('batch shape and dtype')
print(batch_toTensor.shape)
print(batch_toTensor.dtype)

# creates model and loads weights
model = PHOSCnet_temporalpooling()
model.load_state_dict(torch.load(os.path.join('PHOSCnet_temporalpooling', 'epoch31.pt')))

# creates optimizer
optimizer = Adam(model.parameters(), lr=0.0001)

# passes through the batch of images
y_hat = model(batch_toTensor)
y_hat_man = model(batch_manual)

print('y_hat vectors shape')
print(y_hat['phos'].shape, y_hat['phos'].dtype)
print(y_hat['phoc'].shape, y_hat['phoc'].dtype)

# calculates the different losses
phos_loss = F.mse_loss(y_hat['phos'], batch_y_s)
phoc_loss = F.cross_entropy(y_hat['phoc'], batch_y_c)
phoc_loss_binary = F.binary_cross_entropy(y_hat['phoc'], batch_y_c)

print('MSE loss, CE loss, BCE loss')
print(phos_loss)
print(phoc_loss)
print(phoc_loss_binary)

# total loss
loss = phos_loss + phoc_loss
print('total loss')
print(loss)

# backpropagate error
loss.backward()

# adjusting weight according to backpropagation
optimizer.step()

print('backpropagating and steping was succesfull')

print()
# should give 1 or near 1 for all however doesn't... there must be differences between the original toTensor and the way I do it manually.
print(torch.cosine_similarity(y_hat['phos'], y_hat_man['phos']))
print(torch.cosine_similarity(y_hat['phoc'], y_hat_man['phoc']))

