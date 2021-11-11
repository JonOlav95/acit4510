import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_cleaning import *
from plotting import *
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from collections import Counter


def roc_evaluate(clf, x, y_true, label=None):
    y_proba = sklearn.model_selection.cross_val_predict(clf, x, y_true, cv=3, method="predict_proba")
    y_score = y_proba[:, 1]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    plot_roc_curve(tpr, fpr, label)


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx
    else:
        return idx


def my_cross_val(model, x_train, y_train, desired_recall=0.7):
    skfolds = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    avg_threshold = []
    avg_f1 = []

    for train_index, test_index in skfolds.split(x_train, y_train):
        clone_model = clone(model)
        x_train_folds = x_train.iloc[train_index]
        y_train_folds = y_train.iloc[train_index]
        x_test_fold = x_train.iloc[test_index]
        y_test_fold = y_train.iloc[test_index]

        clone_model.fit(x_train_folds, y_train_folds.values.ravel())
        y_pred = clone_model.predict(x_test_fold)
        y_proba = clone_model.predict_proba(x_test_fold)

        f1 = sklearn.metrics.f1_score(y_test_fold, y_pred, average="macro")
        avg_f1.append(f1)

        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test_fold, y_proba[:, 1])
        index = find_nearest(precision, desired_recall)
        avg_threshold.append(thresholds[index])

    return np.mean(avg_f1), np.mean(avg_threshold)


def backward_stepwise_selection(model, x_train, y_train):
    length = len(x_train.columns) - 1
    f1_scores = []

    # Iterate through all the features
    for j in range(length):
        highest_f1 = -1

        for i in range(0, len(x_train.columns)):
            x_step = x_train.drop(x_train.columns[i], axis=1)
            clone_model = clone(model)
            f1 = sklearn.model_selection.cross_val_score(clone_model, x_step, y_train.values.ravel(), cv=6,
                                                         scoring="f1_macro")
            f1 = np.mean(f1)

            f1_2, tmpp = my_cross_val(model, x_step, y_train)

            if f1 > highest_f1:
                highest_f1 = f1
                worst_feature = x_train.columns[i]

        print("Permanent removing " + str(worst_feature))
        print("Best score: " + str(highest_f1))
        x_train.drop(worst_feature, axis=1, inplace=True)
        f1_scores.append(highest_f1)

    plt.plot(list(reversed(f1_scores)))
    plt.xlabel("Number of features")
    plt.ylabel("F1 Score")
    plt.show()


def tmp(x_train, y_train, x_test, y_test):
    model = LogisticRegression(random_state=0, max_iter=10000)
    model.fit(x_train, y_train)
    print("-" * 20 + "logistic regiression" + "-" * 20)
    y_pred = model.predict(x_test)
    arg_test = {"y_true": y_test, "y_pred": y_pred}
    print(sklearn.metrics.confusion_matrix(**arg_test))
    print(sklearn.metrics.classification_report(**arg_test))


def test_model(model, x_train, x_test, y_train, y_test):
    f1, threshold = my_cross_val(model, x_train, y_train)
    model.fit(x_train, y_train.values.ravel())

    y_pred = model.predict_proba(x_test)
    y = [1 if value >= threshold else 0 for value in y_pred[:, 1]]

    # TN  FN
    # FP  TP
    arg_test = {"y_true": y_test, "y_pred": y}
    cf_matrix = sklearn.metrics.confusion_matrix(**arg_test)

    print(cf_matrix)
    print(sklearn.metrics.classification_report(**arg_test))

    return cf_matrix


def future_risk_one(model, normalizer, y, x_test, y_test):
    first = np.array(y)
    second = y_test.values.ravel()

    tp_indices = np.where(np.logical_and(first == 1, second == 1))[0]
    one_record = y_test.iloc[[tp_indices[0]]]
    index_name = one_record.index.values
    record = x_test.loc[[index_name[0]]]

    current_proba = model.predict_proba(record)[0, 1].round(2)

    record["age"] = normalizer.unormalize("age", record["age"])
    record["age"] += 5
    record["age"] = normalizer.normalize("age", record["age"])

    future_proba = model.predict_proba(record)[0, 1].round(2)

    print("current risk: " + str(current_proba))
    print("risk in 5 years: " + str(future_proba))


def add_years(df, normalizer, years=5):
    df["age"] = normalizer.unormalize("age", df["age"])
    df["age"] += years
    df["age"] = normalizer.normalize("age", df["age"])


def main():
    dataframe = pd.read_csv("dataset.csv")

    dataframe, normalizer = clean_data(dataframe)
    x_train, x_test, y_train, y_test = new_split(dataframe)

    lr_model = LogisticRegression(random_state=42, max_iter=20000)
    ranfor_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gnb_model = DecisionTreeClassifier()
    svc_model = SVC(kernel='sigmoid', gamma='scale')
    knn_model = KNeighborsClassifier()
    gaus_model = GaussianNB()
    kmeans_model = KMeans(n_clusters=2, n_init=10, random_state=42)

    cf_matrix_now = test_model(lr_model, x_train, x_test, y_train, y_test)
    add_years(x_test, normalizer)
    cf_matrix_future = test_model(lr_model, x_train, x_test, y_train, y_test)

    fn_diff = cf_matrix_future[0][1] - cf_matrix_now[0][1]
    tp_diff = cf_matrix_future[1][1] - cf_matrix_now[1][1]

    #future_risk_one(lr_model, normalizer, y, x_test, y_test)
    # backward_stepwise_selection(lr_model, x_train, y_train)


if __name__ == '__main__':
    main()
