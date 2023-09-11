import os
import random

import pandas as pd

import augmentation

import cv2 as cv

from tqdm import tqdm

import shutil
import numpy as np
from PIL import Image


def count_words(df: pd.DataFrame) -> dict:
    word_counter = {}

    for word in df['Word']:
        if not word in word_counter.keys():
            word_counter[word] = 1
        else:
            word_counter[word] += 1

    return word_counter


def validate_dataset(df: pd.DataFrame, path: str):
    for img in tqdm(df['Image']):
        if not os.path.exists(os.path.join(path, img)):
            raise FileNotFoundError(os.path.join(path, img))


def copy_df_content_to_folder(df: pd.DataFrame, in_path: str, out_path: str):
    for file in tqdm(df['Image']):
        img = cv.imread(os.path.join(in_path, file))

        img = augmentation.resize_img(img)

        cv.imwrite(os.path.join(out_path, file), img)


def is_augmented(name: str):
    startIndex = name.index('.png')

    name = name[:startIndex]

    for ch in name:
        if ch == '-':
            return False

    return True


def augment_word_class(df: pd.DataFrame,
                       word: str,
                       current_n: int,
                       in_folder_path: str,
                       out_folder_path: str,
                       total_n: int,
                       noise_variability:int=30,
                       max_shear_factor:int=2)->pd.DataFrame:

    iterations = total_n-current_n

    if iterations <= 0:
        return df

    df_sub = df[df['Word'] == word]

    for i in range(iterations):
        img_i = random.randint(0, len(df_sub)-1)

        img_fname = df_sub.iloc[img_i, 0]

        shearing_factor = random.random() * max_shear_factor

        img = cv.imread(f'{in_folder_path}/{img_fname}')

        img = augmentation.resize_img(img)
        img = augmentation.shear_image(img, shearing_factor)
        img = augmentation.resize_img(img)
        img = augmentation.noise_image(img, noise_variability)

        cv.imwrite(f'{out_folder_path}/{img_fname}_aug{i}.png', img)

        new_sample = pd.DataFrame([{'Image': f'{img_fname}_aug{i}.png', 'Word': word}])

        df = pd.concat([df, new_sample])

    return df


def augment_index(df: pd.DataFrame,
                       index: int,
                       in_folder_path: str,
                       out_folder_path: str,
                       total_n: int,
                       noise_variability:int=30,
                       max_shearx_factor:int=1,
                       max_sheary_factor:int=0.05
                       )->pd.DataFrame:

    
    for i in range(total_n):
        img_fname = df.iloc[index, 0]
        img_word = df.iloc[index, 1]


        augs = [augmentation.noise_image, augmentation.shear_x,
            augmentation.shear_y, augmentation.random_perspective,
            augmentation.erode, augmentation.dialate,
            augmentation.blur, augmentation.sharpness
        ]

        img = Image.open(f'{in_folder_path}/{img_fname}')
        prev = None

        for a in range(np.random.randint(1, 3)):
            op_n = np.random.randint(0, len(augs))

            while op_n == prev:
                op_n = np.random.randint(0, 8)

            prev = op_n

            if op_n == 0:
                img = augmentation.noise_image(img, noise_variability)
            elif op_n == 1:
                factor = np.random.uniform(low=-max_shearx_factor, high=max_shearx_factor)
                img = augmentation.shear_x(img, factor)
            elif op_n == 2:
                factor = np.random.uniform(low=-max_sheary_factor, high=max_sheary_factor)
                img = augmentation.shear_y(img, factor)
            elif op_n == 3:
                img = augmentation.random_perspective(img)
            elif op_n == 4:
                img = augmentation.erode(img, 1)
            elif op_n == 5:
                img = augmentation.dialate(img, 1)
            elif op_n == 6:
                factor = np.random.uniform(1, 2)
                img = augmentation.blur(img, factor)
            elif op_n == 7:
                factor = np.random.uniform(5, 10)
                img = augmentation.sharpness(img, factor)
        



        # shearing_factor_ = np.random.uniform(low=)

        # img = cv.imread(f'{in_folder_path}/{img_fname}')

        # img = augmentation.augment_image(img, shearing_factor, noise_variability)

        # cv.imwrite(f'{out_folder_path}/{img_fname[:img_fname.index(".png")]}_aug{i}.png', img)

        img.save(f'{out_folder_path}/{img_fname[:img_fname.index(".png")]}_aug{i}.png')

        new_sample = pd.DataFrame([{'Image': f'{img_fname[:img_fname.index(".png")]}_aug{i}.png', 'Word': img_word}])

        df = pd.concat([df, new_sample])

    return df


def main():
    random.seed(1)

    in_folder = r'image_data/norwegianDataNewGZSL/0/norwegian9000_train_0'
    out_folder = r'image_data/norwegianDataNewGZSL/0/norwegian9000_train_0_superaug20'

    image_per_word = 50
    image_per_index = 20

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    df = pd.read_csv(f'{in_folder}.csv')
    word_dict = count_words(df)

    copy_df_content_to_folder(df, in_folder, out_folder)
    validate_dataset(df, out_folder)

    # for word in tqdm(word_dict.keys()):
    #     df = augment_word_class(
    #         df, word, word_dict[word], in_folder, out_folder, image_per_word)

    for i in tqdm(range(len(df))):
        if is_augmented(df.iloc[i, 0]):
            continue
        
        df = augment_index(df, i, in_folder, out_folder, image_per_index)

    
    df.to_csv(f'{out_folder}.csv', index=False)

    validate_dataset(df, out_folder)


if __name__ == '__main__':
    main()
