import os
import cv2 as cv
from tqdm import tqdm


def main():
    splits = [
        r'image_data/norwegianDataNewGZSL/3/norwegian9000_valid_3',
        r'image_data/norwegianDataNewGZSL/3/norwegian9000_valid_3_resized'
    ]

    os.makedirs(splits[1], exist_ok=True)

    in_folder_path = splits[0]
    output_folder_path = splits[1]

    for img_name in tqdm(os.listdir(in_folder_path)):
        img = cv.imread(os.path.join(in_folder_path, img_name))

        img = cv.resize(img, (250, 50))

        cv.imwrite(os.path.join(output_folder_path, img_name), img)


if __name__ == '__main__':
    main()