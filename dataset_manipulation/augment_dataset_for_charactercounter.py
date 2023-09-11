from ast import Raise
import os
import random

import pandas as pd

import augmentation

import cv2 as cv

from tqdm import tqdm

import shutil

def find_max_length_word(df: pd.DataFrame)->int:
    mx_len = 0
    mx_word = ''
    for word in df['Word']:
        if mx_len < len(word):
            mx_len = len(word)
            mx_word = word
    return mx_len, mx_word

def count_words(df: pd.DataFrame) -> dict:
    word_counter = {}

    for word in df['Word']:
        if not word in word_counter.keys():
            word_counter[word] = 1
        else:
            word_counter[word] += 1

    return word_counter

def count_words_per_length(df: pd.DataFrame):
    word_count = count_words(df)
    samples_per_len = {}

    for i in range(1, find_max_length_word(df)[0]+1):
        samples_per_len[i] = 0

    for key in word_count.keys():
        word_len = len(key)
        
        samples_per_len[word_len] += word_count[key]

    return samples_per_len


def validate_dataset(df: pd.DataFrame, path: str):
    for img in tqdm(df['Image']):
        if not os.path.exists(os.path.join(path, img)):
            raise FileNotFoundError(os.path.join(path, img))


def copy_df_content_to_folder(df: pd.DataFrame, in_path: str, out_path: str):
    for file in tqdm(df['Image']):
        img = cv.imread(os.path.join(in_path, file))

        img = augmentation.resize_img(img)

        cv.imwrite(os.path.join(out_path, file), img)


def get_desired_df(df, l):
    Image = []
    Word = []
    
    for i in range(len(df)):
        # check if row has word of right length and the image isn't already augmentated
        if len(df.iloc[i, 1]) == l and not df.iloc[i, 0][0:df.iloc[i, 0].index('.png')].isnumeric():
            Image.append(df.iloc[i, 0])
            Word.append(df.iloc[i, 1])

    return pd.DataFrame(list(zip(Image, Word)),
               columns=['Image', 'Word'])

    




def augment_word_len(df: pd.DataFrame,
                       l: str,
                       current_n: int,
                       total_n: int,
                       in_folder_path: str,
                       out_folder_path: str,
                       noise_variability:int=30,
                       max_shear_factor:int=2)->pd.DataFrame:

    iterations = total_n-current_n

    if iterations <= 0:
        raise Exception('Number of total images is higher then requested amount!')

    df_sub = get_desired_df(df, l)

    Image_lst = []
    Word_lst = []

    for i in tqdm(range(iterations)):
        img_i = random.randint(0, len(df_sub)-1)

        img_fname = df_sub.iloc[img_i, 0]
        word = df_sub.iloc[img_i, 1]

        shearing_factor = random.random() * max_shear_factor

        img = cv.imread(f'{in_folder_path}/{img_fname}')

        img = augmentation.resize_img(img)
        img = augmentation.shear_image(img, shearing_factor)
        img = augmentation.resize_img(img)
        img = augmentation.noise_image(img, noise_variability)

        cv.imwrite(f'{out_folder_path}/{img_fname}_aug{i}.png', img)

        Image_lst.append(f'{img_fname}_aug{i}.png')
        Word_lst.append(word)

    df_new = pd.DataFrame(list(zip(Image_lst, Word_lst)),
               columns=['Image', 'Word'])

    return df_new



def main():
    random.seed(1)

    in_folder = r'image_data/IAM_Data/IAM_train'
    out_folder = r'image_data/IAM_Data/IAM_train_charcount_balance1'

    number_images_per_len = 18_000
    start_len = 8
    end_len = 17

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    df = pd.read_csv(f'{in_folder}.csv')
    word_len_dict = count_words_per_length(df)

    copy_df_content_to_folder(df, in_folder, out_folder)
    validate_dataset(df, out_folder)

    df_next = pd.DataFrame(columns=['Image', 'Word'])

    for l in range(start_len, (end_len+1)):
        df_new = augment_word_len(
            df, l, word_len_dict[l], number_images_per_len, in_folder, out_folder)

        df_next = pd.concat([df_next, df_new], ignore_index=True)

    df = pd.concat([df, df_next])
    
    df.to_csv(f'{out_folder}.csv', index=False)

    validate_dataset(df, out_folder)


if __name__ == '__main__':
    main()
