import os
import shutil
import pandas as pd
from tqdm import tqdm


def get_label_from_filename(filename: str) -> str:
    label = ''
    i = filename.index(']_') + 2

    while filename[i] != '_':
        label += filename[i]

        i += 1

    return label


def is_word(word) -> bool:
    for ch in word:
        if not ch.isalpha():
            return False

    return True


def trim_none_characters_list_format(image_path: str) -> dict:
    files_list = []

    for file in os.listdir(image_path):
        word = get_label_from_filename(file)

        if is_word(word):
            files_list.append([file, word])

    return files_list


def main():
    # original iamsplit, count of images
    print('train, test, validate, total')
    print(len(os.listdir(r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#')), end=', ')
    print(len(os.listdir(r'image_data/IamSplit/data/iamOffValCrops#07-2022-07-7#')), end=', ')
    print(len(os.listdir(r'image_data/IamSplit/data/iamOffTestCrops#07-2022-07-7#')), end=', ')
    print(len(os.listdir(r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#')) +
          len(os.listdir(r'image_data/IamSplit/data/iamOffValCrops#07-2022-07-7#')) +
          len(os.listdir(r'image_data/IamSplit/data/iamOffTestCrops#07-2022-07-7#')))

    data_splits = [
        r'image_data/IamSplit/data/iamOffTrainCrops#07-2022-07-7#',
        r'image_data/IamSplit/data/iamOffValCrops#07-2022-07-7#',
        r'image_data/IamSplit/data/iamOffTestCrops#07-2022-07-7#'
    ]

    splits = [
        'train',
        'valid',
        'test'
    ]

    trimmed_path = os.path.join(
        os.getcwd(), 'image_data', 'IamSplit', 'trimmed_data')
    if not os.path.exists(trimmed_path):
        os.mkdir(trimmed_path)

    for i in range(len(data_splits)):
        df = pd.DataFrame(trim_none_characters_list_format(
            data_splits[i]), columns=['Image', 'Word'])

        print(df)
        print(df.columns)

        if not os.path.exists(os.path.join(trimmed_path, splits[i])):
            os.mkdir(os.path.join(trimmed_path, splits[i]))

        for file in tqdm(df['Image']):
            shutil.copyfile(os.path.join(data_splits[i], file), os.path.join(
                trimmed_path, splits[i], file))

        df.to_csv(os.path.join(trimmed_path, f'{splits[i]}.csv'), index=False)



if __name__ == '__main__':
    main()
