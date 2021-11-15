from imblearn.over_sampling import SMOTE
import pandas as pd
import sklearn
from sklearn.utils import shuffle

from normalizer import Normalizer


def one_hot_encoding(df):
    for col in df.columns:
        if df[col].dtypes == object:
            n_uniques = df[col].unique()

            if len(n_uniques) == 2:
                df[col] = df[col].replace([n_uniques[0]], 0)
                df[col] = df[col].replace([n_uniques[1]], 1)
                continue

            name = df[col].name
            df = pd.concat([df.drop(name, axis=1), pd.get_dummies(df[name], prefix=name)], axis=1)

    return df


def clean_data(df):
    df.fillna(df.mean(), inplace=True)
    #df = shuffle(df)

    # Drop the ID Column
    df.pop("id")

    # Drop the "other" gender rows as there is only one
    df = df[df.gender != "Other"]
    df = pd.get_dummies(df)

    normalizer = Normalizer()

    df["age"] = normalizer.normalize("age", df["age"])
    df["bmi"] = normalizer.normalize("bmi", df["bmi"])
    df["avg_glucose_level"] = normalizer.normalize("avg_glucose_level", df["avg_glucose_level"])

    return df, normalizer


def new_split(df):
    y = df[["stroke"]]
    X = df.drop("stroke", axis=1)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, stratify=y)

    smote = SMOTE(sampling_strategy="minority")
    oversample_x, oversample_y = smote.fit_resample(x_train, y_train)
    print("Shape of X: {}".format(x_train.shape))
    print("Shape of y: {}".format(y_train.shape))

    return x_train, x_test, y_train, y_test, oversample_x, oversample_y
