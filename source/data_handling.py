import pandas as pd
from sklearn.utils import shuffle


def one_hot_encoding(df):
    for col in df.columns:
        if df[col].dtypes == object:
            name = df[col].name
            df = pd.concat([df.drop(name, axis=1), pd.get_dummies(df[name], prefix=name)], axis=1)

    return df


def split_data(df):
    df.fillna(df.mean(), inplace=True)
    df = shuffle(df)
    df.pop("id")
    df = df[df.gender != "Other"]
    df = one_hot_encoding(df)

    df_true = df[df["stroke"] == 1]
    df_false = df[df["stroke"] == 0]

    n1 = df_true.shape[0]
    x1 = int(n1 / 10)

    n2 = df_false.shape[0]
    x2 = int(n2 / 10)

    val_data = pd.concat([df_true[:x1], df_false[:x2]])
    test_data = pd.concat([df_true[x1:(x1 * 2)], df_false[x2:(x2 * 2)]])
    train_data = pd.concat([df_true[(x1 * 2):], df_false[(x2 * 2):]])

    val_data = shuffle(val_data)
    test_data = shuffle(test_data)
    train_data = shuffle(train_data)

    return train_data, test_data, val_data
