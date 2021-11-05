import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from data_handling import *
from evaluation import *
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def cast_by_threshold(y, threshold):
    for i in range(len(y)):
        if y[i] > threshold:
            y[i] = 1
        else:
            y[i] = 0


def my_cross_val(model, x_train, y_train):
    skfolds = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    threshold = 0.18
    avg_f1 = 0

    for train_index, test_index in skfolds.split(x_train, y_train):
        clone_model = clone(model)
        x_train_folds = x_train.iloc[train_index]
        y_train_folds = y_train.iloc[train_index]
        x_test_fold = x_train.iloc[test_index]
        y_test_fold = y_train.iloc[test_index]

        clone_model.fit(x_train_folds, y_train_folds)
        y_pred = clone_model.predict_proba(x_test_fold)
        y_pred = y_pred[:, 1]

        cast_by_threshold(y_pred, threshold)

        f1 = sklearn.metrics.f1_score(y_test_fold, y_pred, average="macro")

        avg_f1 += f1

    avg_f1 /= 6
    return avg_f1


def roc_evaluate(clf, x, y_true, label=None):
    y_proba = sklearn.model_selection.cross_val_predict(clf, x, y_true, cv=3, method="predict_proba")
    y_score = y_proba[:, 1]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    plot_roc_curve(tpr, fpr, label)


def precision_recall_evaluate(clf, x, y_true, label=None):
    y_proba = clf.predict_proba(x)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_proba[:, 1])
    precision_recall_plot(precision, recall, thresholds, label)


def backward_stepwise_selection(x_train, y_train):
    model = LogisticRegression(random_state=0, max_iter=10000)

    highest_f1 = my_cross_val(model, x_train, y_train)
    worst_feature = x_train.columns[0]
    length = len(x_train.columns) - 2

    for j in range(length):
        for i in range(len(x_train.columns)):
            clone_model = clone(model)

            x_step = x_train.drop(x_train.columns[i], axis=1)
            f1 = my_cross_val(clone_model, x_step, y_train)

            if f1 > highest_f1:
                worst_feature = x_train.columns[i]

        print("Permanent removing " + str(worst_feature))
        x_train.drop(worst_feature, axis=1, inplace=True)
        clone_model = clone(model)

        f1 = my_cross_val(clone_model, x_train, y_train)
        print("F1: " + str(f1))


def main():
    dataframe = pd.read_csv("dataset.csv")
    x_train, y_train, x_test, y_test = split_data(dataframe)

    lr_clf = LogisticRegression(random_state=0, max_iter=10000)
    forest_clf = RandomForestClassifier(random_state=42)

    backward_stepwise_selection(x_train, y_train)
    #my_cross_val(lr_clf, x_train, y_train)
    '''
    lr_clf.fit(x_train, y_train)
    forest_clf.fit(x_train, y_train)

    precision_recall_evaluate(forest_clf, x_train, y_train, "Random Forest")
    precision_recall_evaluate(lr_clf, x_train, y_train, "Logistic Regression")

    roc_evaluate(forest_clf, x_train, y_train, "Random Forest")
    roc_evaluate(lr_clf, x_train, y_train, "Logistic Regression")
    plt.show()

    score = sklearn.model_selection.cross_val_score()
    '''


if __name__ == '__main__':
    main()
