import pandas as pd

def create_sub_df(df: pd.DataFrame, words: set):
    new_df = pd.DataFrame(columns=['Image', 'Word'])

    for word in words:
        word_df = df[df['Word'] == word]
        new_df = pd.concat([new_df, word_df])

    return new_df

def main():
    splits = [
        r'image_data/IamSplit/augmented_data/train',
        r'image_data/IamSplit/augmented_data/valid',
        r'image_data/IamSplit/augmented_data/test',
        ]

    df_train = pd.read_csv(f'{splits[0]}.csv')
    df_valid = pd.read_csv(f'{splits[1]}.csv')
    df_test = pd.read_csv(f'{splits[2]}.csv')

    train_words = df_train['Word'].unique()
    valid_words = set(df_valid['Word'].unique())
    test_words = set(df_test['Word'].unique())

    print(len(valid_words))
    print(len(test_words))

    print(type(valid_words))

    valid_intersect_words = valid_words.intersection(train_words)
    valid_diff_words = valid_words.difference(train_words)
    
    test_intersect_words = test_words.intersection(train_words)
    test_diff_words = test_words.difference(train_words)

    valid_seen_df = create_sub_df(df_valid, valid_intersect_words)
    valid_unseen_df = create_sub_df(df_valid, valid_diff_words)

    test_seen_df = create_sub_df(df_test, test_intersect_words)
    test_unseen_df = create_sub_df(df_test, test_diff_words)

    valid_seen_df.to_csv(f'{splits[1]}_seen.csv', index=False)
    valid_unseen_df.to_csv(f'{splits[1]}_unseen.csv', index=False)
    
    test_seen_df.to_csv(f'{splits[2]}_seen.csv', index=False)
    test_unseen_df.to_csv(f'{splits[2]}_unseen.csv', index=False)



    



if __name__ == '__main__':
    main()