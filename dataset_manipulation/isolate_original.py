from augment_dataset import is_augmented, copy_df_content_to_folder, validate_dataset
from tqdm import tqdm

import pandas as pd
import augmentation
import cv2 as cv
import os


def main():
    in_path = 'image_data/GW_Data/CV4_train'
    out_path = 'image_data/GW_Data/CV4_train_noaug'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    df = pd.read_csv(f'{in_path}.csv')

    new_df = pd.DataFrame(columns=['Image', 'Word'])

    for i in tqdm(range(len(df))):
        if not is_augmented(df.iloc[i, 0]):
            img = cv.imread(os.path.join(in_path, df.iloc[i, 0]))

            img = augmentation.resize_img(img)

            cv.imwrite(os.path.join(out_path, df.iloc[i, 0]), img)

            new_sample = pd.DataFrame([{'Image': f'{df.iloc[i, 0]}', 'Word': df.iloc[i, 1]}])

            new_df = pd.concat([new_df, new_sample])


    validate_dataset(new_df, out_path)
    new_df.to_csv(f'{out_path}.csv', index=False)
    

            


if __name__ == '__main__':
    main()