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

    # Drop the ID Column
    df.pop("id")

    # Drop the "other" gender rows as there is only one
    df = df[df.gender != "Other"]
    df = one_hot_encoding(df)

    df_true = df[df["stroke"] == 1]
    df_false = df[df["stroke"] == 0]

    x1 = int(df_true.shape[0] / 10)

    test_data = pd.concat([df_true[:x1], df_false[:x1]])
    train_data = pd.concat([df_true[x1:], df_false[x1:]])

    test_data = shuffle(test_data)
    train_data = shuffle(train_data)

    y_train = train_data["stroke"]
    x_train = train_data.drop("stroke", axis=1)

    y_test = test_data["stroke"]
    x_test = test_data.drop("stroke", axis=1)

    return x_train, y_train, x_test, y_test
