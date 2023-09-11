import os
import random

import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from timm.data.auto_augment import shear_x
from timm.data.random_erasing import RandomErasing
from torchvision import transforms
from torchvision.transforms import RandomPerspective
import torch


# modified version of: https://github.com/anuj-rai-23/PHOSC-Zero-Shot-Word-Recognition/blob/main/aug_images.py
def noise_image(img, variability):
    img = np.int32(np.asarray(img))

    deviation = variability*random.random()

    noise = np.int32(np.random.normal(0, deviation, img.shape))

    img += noise
    img = np.uint8(np.clip(img, 0., 255.))
    
    return Image.fromarray(img)


def shear_x(img, factor):
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), fillcolor=(255, 255, 255))


def shear_y(img, factor):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), fillcolor=(255, 255, 255))


def resize_img(img):
    img = cv.resize(img, (250, 50)).copy()

    return img

def gray_scale_img(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img

def threshold_image(img):
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

    # black pixel to white and white to black
    for rows in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[rows, col] == 255:
                img[rows, col] = 0
            else:
                img[rows, col] = 255

    return img

def random_perspective(img):
    return RandomPerspective(0.5, p=1, fill=255)(img)

def erode(img, cycles):
     for _ in range(cycles):
          img = img.filter(ImageFilter.MinFilter(3))
     return img

def dialate(img, cycles):
     for _ in range(cycles):
          img = img.filter(ImageFilter.MaxFilter(3))
     return img

def sharpness(img, factor):
    return ImageEnhance.Sharpness(img).enhance(factor)

def blur(img, factor):
    img = img.filter(ImageFilter.GaussianBlur(radius=factor))
    return img

def random_erase(img):
    x  = transforms.ToTensor()(img)
    random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
    x = transforms.ToPILImage()(random_erase(x))

    print(x)

    return x


def main():
    img_path = 'image_data/IAM_Data/IAM_train_noaug/a01-000u-00-06.png'

    img = Image.open(img_path)

    # img_shear = shear_x(img, 1) # [1, -1]
    img_shear = shear_y(img, 0.05) # [-0.05, 0.05]

    img_shear.save('image_data/shear_test.png')

    img_pers = random_perspective(img)
    img_pers.save('image_data/perspective_test.png')

    img_noise = noise_image(img, 30)
    img_noise.save('image_data/noise_test.png')

    img_erode = erode(img, 1)
    img_erode.save('image_data/erode_test.png')

    img_dialate = dialate(img, 1)
    img_dialate.save('image_data/dialate_test.png')

    img_sharp = sharpness(img, 10)
    img_sharp.save('image_data/sharper_test.png')

    img_blur = blur(img, 3)
    img_blur.save('image_data/blur_test.png')

    img_erase = random_erase(img)
    img_erase.save('image_data/erase_test.png')


    print(np.random.uniform(low=-1, high=0))






    # print('train, test, validate, total')
    # print(len(os.listdir(r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#')), end=', ')
    # print(len(os.listdir(r'image_data/IamSplit/data/iamOffValCrops#07-2022-07-7#')), end=', ')
    # print(len(os.listdir(r'image_data/IamSplit/data/iamOffTestCrops#07-2022-07-7#')), end=', ')
    # print(len(os.listdir(r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#')) + 
    #     len(os.listdir(r'image_data/IamSplit/data/iamOffValCrops#07-2022-07-7#')) + 
    #     len(os.listdir(r'image_data/IamSplit/data/iamOffTestCrops#07-2022-07-7#')))

    # img_path = r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#/a01-000u.png_16454_[1405, 1140, 1469, 1175]_a_#07-2022-07-7#.png'
    # img_path = r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#/a01-000u.png_16443_[395, 932, 836, 1032]_nominating_#07-2022-07-7#.png'

    # img = cv.imread(img_path)

    # cv.imwrite(r'image_data/test.png', img)

    # img = cv.resize(img, (250, 50))

    # img_sheared = shear_image(img, 2)

    # img_sheared = cv.resize(img_sheared, (250, 50))

    # cv.imwrite(r'image_data/sheared_test.png', img_sheared)

    # img_noise = noise_image(img_sheared, 30)

    # cv.imwrite(r'image_data/noise_test.png', img_noise)

    # img_gray = gray_scale_img(img)

    # cv.imwrite(r'image_data/greay_test.png', img_gray)

    # img_cc = threshold_image(img_gray)

    # cv.imwrite(r'image_data/threshold_test.png', img_cc)




if __name__ == "__main__":
    main()