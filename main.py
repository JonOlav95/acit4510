import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


if __name__ == '__main__':
    dataframe = pd.read_csv("dataset.csv")
    print(dataframe.shape)
