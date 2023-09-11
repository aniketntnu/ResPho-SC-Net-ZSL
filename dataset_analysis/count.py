import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

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

def main():
    df_train = pd.read_csv('image_data/IAM_Data/IAM_train.csv')
    df_valid = pd.read_csv('image_data/IAM_Data/IAM_valid.csv')
    df_test_seen = pd.read_csv('image_data/IAM_Data/IAM_test_seen.csv')
    df_test_unseen = pd.read_csv('image_data/IAM_Data/IAM_test_unseen.csv')

    print(df_train['Word'])

    print(find_max_length_word(df_train))
    print(find_max_length_word(df_valid))
    print(find_max_length_word(df_test_seen))
    print(find_max_length_word(df_test_unseen))

    count_word_lens = count_words_per_length(df_train)
    print(count_word_lens)
    print(count_word_lens.keys())
    print(count_word_lens.values())

    plt.bar(count_word_lens.keys(), list(count_word_lens.values()))
    plt.xticks(range(1, 18))
    plt.savefig('count_words_per_len.png')


if __name__ == '__main__':
    main()