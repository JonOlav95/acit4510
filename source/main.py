import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
import sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def one_hot_encoding(df):
    for col in df.columns:
        if df[col].dtypes == object:
            name = df[col].name
            df = pd.concat([df.drop(name, axis=1), pd.get_dummies(df[name], prefix=name)], axis=1)

    return df


def split_data(df):
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


def main():
    dataframe = pd.read_csv("dataset.csv")
    train_data, test_data, val_data = split_data(dataframe)
    train_data.fillna(dataframe.mean(), inplace=True)
    test_data.fillna(dataframe.mean(), inplace=True)

    for col in train_data:
        print(col)

    print(train_data.describe().transpose()[["mean", "std"]])

    X = train_data.drop("stroke", axis=1)

    y = train_data["stroke"]
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)

    x_new = test_data.drop("stroke", axis=1)
    y_new = test_data[["stroke"]]

    y_predict = clf.predict_proba(x_new)
    # y_predict = y_predict[y_predict[:,1].argsort()]

    decision_boundary = 0.2
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for i in range(len(y_predict)):
        print("PRED: " + str(y_predict[i][1]) + "\tREAL: " + str(y_new.iloc[i][0]))

        if y_predict[i][1] >= decision_boundary:

            if y_new.iloc[i][0] == 1:
                true_positive += 1
            else:
                false_positive += 1

        else:
            if y_new.iloc[i][0] == 1:
                false_negative += 1
            else:
                true_negative += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = (2 * precision * recall) / (precision + recall)

    print("true positive: " + str(true_positive))
    print("true negative: " + str(true_negative))
    print("false positive: " + str(false_positive))
    print("false negative: " + str(false_negative))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1: " + str(f1))

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_new, y_predict[:, 1])
    # retrieve probability of being 1(in second column of probs_y)
    pr_auc = sklearn.metrics.auc(recall, precision)

    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0, 1])

    plt.show()


if __name__ == '__main__':
    # physical_devices = tf.config.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()
