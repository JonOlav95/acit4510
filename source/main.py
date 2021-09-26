import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("stroke")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def split_data(dataframe):
    dataframe = shuffle(dataframe)
    n = dataframe.shape[0]

    x = n / 10

    val_data = dataframe[0:x]
    test_data = dataframe[x:(2 * x)]
    train_data = dataframe[(2 * x):]

    print("x")


def main():
    dataframe = pd.read_csv("dataset.csv")
    split_data(dataframe)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    main()

