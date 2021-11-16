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
import statsmodels.api as sm


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


def my_cross_val(model, x_train, y_train):
    skfolds = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    avg_threshold = []

    for train_index, test_index in skfolds.split(x_train, y_train):
        clone_model = clone(model)
        x_train_folds = x_train.iloc[train_index]
        y_train_folds = y_train.iloc[train_index]
        x_test_fold = x_train.iloc[test_index]
        y_test_fold = y_train.iloc[test_index]

        clone_model.fit(x_train_folds, y_train_folds.values.ravel())

        y_pred = clone_model.predict_proba(x_test_fold)
        y_pred = y_pred[:, 1]

        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test_fold, y_pred)
        fscore = (2 * precision * recall) / (precision + recall)
        max_fscore = np.nanargmax(fscore)
        t = thresholds[max_fscore]
        largest_f1 = fscore[max_fscore]
        avg_threshold.append(t)
        precplot(precision, recall, thresholds)

        decision = (y_pred >= t).astype('int')
        arg_test = {"y_true": y_test_fold, "y_pred": decision}

        #print(sklearn.metrics.confusion_matrix(**arg_test))
        #print(sklearn.metrics.classification_report(**arg_test))

    return np.mean(avg_threshold)


def backward_stepwise_selection(model, x_train, y_train):
    length = len(x_train.columns) - 1
    f1_scores = []

    # Iterate through all the features
    for j in range(length):
        highest_f1 = -1

        for i in range(0, len(x_train.columns)):
            x_step = x_train.drop(x_train.columns[i], axis=1)
            f1 = my_cross_val(model, x_step, y_train)

            f1 = np.mean(f1)

            if f1 > highest_f1:
                highest_f1 = f1
                worst_feature = x_train.columns[i]

        print("Permanent removing " + str(worst_feature))
        print("Best score: " + str(highest_f1))
        x_train.drop(worst_feature, axis=1, inplace=True)
        f1_scores.append(highest_f1)

    plt.plot(range(1, len(f1_scores) + 1), list(reversed(f1_scores)))
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


def test_model_simple(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train.values.ravel())

    y_pred = model.predict(x_test)

    arg_test = {"y_true": y_test, "y_pred": y_pred}
    cf_matrix = sklearn.metrics.confusion_matrix(**arg_test)

    print(cf_matrix)
    print(sklearn.metrics.classification_report(**arg_test))


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


def tmp_forest(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    feature_names = list(x_train.columns.values)
    importances, feature_names = (list(t) for t in zip(*sorted(zip(importances, feature_names))))

    sns.barplot(importances, feature_names)
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.show()


def pvalue(x_train, y_train):
    logit_model = sm.Logit(y_train, x_train)
    result = logit_model.fit(maxiter=1000)
    print(result.summary())
    plot_p_value(result)


def test_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train.values.ravel())
    y = model.predict(x_test)

    # TN  FP
    # FN  TP
    arg_test = {"y_true": y_test, "y_pred": y}

    print(sklearn.metrics.confusion_matrix(**arg_test))
    print(sklearn.metrics.classification_report(**arg_test))


def move_threshold(model, x_train, x_test, y_train, y_test):
    threshold = my_cross_val(model, x_train, y_train)
    model.fit(x_train, y_train.values.ravel())
    y = model.predict_proba(x_test)[:, 1]

    decision = (y >= threshold).astype('int')

    arg_test = {"y_true": y_test, "y_pred": decision}

    print(sklearn.metrics.confusion_matrix(**arg_test))
    print(sklearn.metrics.classification_report(**arg_test))


def main():
    dataframe = pd.read_csv("dataset.csv")

    dataframe, normalizer = clean_data(dataframe)
    x_train, x_test, y_train, y_test, oversample_x, oversample_y = new_split(dataframe)

    # pvalue(x_train, y_train)
    # tmp_forest(x_train, x_test, y_train, y_test)

    lr_model = LogisticRegression(class_weight={0: 1, 1: 20}, max_iter=20000)
    ranfor_model = RandomForestClassifier()
    dc_model = DecisionTreeClassifier()
    svc_model = SVC(probability=True, kernel='sigmoid', gamma='scale')
    # knn_model = KNeighborsClassifier()
    gaus_model = GaussianNB()
    # kmeans_model = KMeans(n_clusters=2, n_init=10, random_state=42)

    #x_train = x_train[["age", "avg_glucose_level", "hypertension", "heart_disease", "bmi"]]
    #x_test = x_test[["age", "avg_glucose_level", "hypertension", "heart_disease", "bmi"]]

    test_model(ranfor_model, x_train, x_test, y_train, y_test)
    #move_threshold(ranfor_model, x_train, x_test, y_train, y_test)
    # add_years(x_test, normalizer)
    # test_model(lr_model, x_train, x_test, y_train, y_test)

    # fn_diff = cf_matrix_future[0][1] - cf_matrix_now[0][1]
    # tp_diff = cf_matrix_future[1][1] - cf_matrix_now[1][1]

    # future_risk_one(lr_model, normalizer, y, x_test, y_test)

    #backward_stepwise_selection(lr_model, x_train, y_train)


if __name__ == '__main__':
    main()
