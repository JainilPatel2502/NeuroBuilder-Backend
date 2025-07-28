def get_split(df, test_ratio):
    test = df.sample(frac=test_ratio, random_state=42)
    train = df.drop(test.index)
    return train, test