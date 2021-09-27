import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("stroke")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    return ds


def string_to_integer(dataframe):
    for col in dataframe.columns:
        data_type = dataframe[col].dtypes

        if data_type == object:
            unique = np.unique(dataframe[col].values)

            for i in range(len(unique)):
                dataframe[col].replace(unique[i], i, inplace=True)


def split_data(dataframe):
    dataframe.pop("id")
    dataframe = shuffle(dataframe)
    string_to_integer(dataframe)

    df_true = dataframe[dataframe["stroke"] == 1]
    df_false = dataframe[dataframe["stroke"] == 0]

    n1 = df_true.shape[0]
    x1 = int(n1 / 10)

    n2 = df_false.shape[0]
    x2 = int(n2 / 10)

    val_data =  pd.concat([df_true[:x1], df_false[:x2]])
    test_data = pd.concat([df_true[x1:(x1 * 2)], df_false[x2:(x2 * 2)]])
    train_data = pd.concat([df_true[(x1 * 2):], df_false[(x2 * 2):]])

    val_data = shuffle(val_data)
    test_data = shuffle(test_data)
    train_data = shuffle(train_data)

    return train_data, test_data, val_data


def main():
    dataframe = pd.read_csv("dataset.csv")
    train_data, test_data, val_data = split_data(dataframe)

    print(train_data.describe().transpose()[["mean", "std"]])



if __name__ == '__main__':
    #physical_devices = tf.config.list_physical_devices("GPU")
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()

