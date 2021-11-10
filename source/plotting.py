import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def precision_recall_plot(clf, x, y_true, label=None):
    y_proba = clf.predict_proba(x)
    y_proba = y_proba[:, 1]
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_proba)
    plt.title(label)

    precision[-2] = 1
    recall[-2] = 0

    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0, 1])

    plt.show()


def plot_roc_curve(tpr, fpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "k--")
    plt.ylabel("True Positive Rate (Recall)")
    plt.xlabel("False Positive Rate")


def plot_corr(df):
    plt.figure(figsize=(12, 10))
    df = df[["age", "ever_married", "heart_disease", "stroke", "bmi", "hypertension"]]
    cor = df.corr().round(2)
    sns.heatmap(cor, annot=True)
    plt.show()


def plot_outlier(df):
    df = df["bmi"].sort_values().tolist()
    sns.displot(df, bins=82, kde=True)
    plt.xlabel("BMI")
    plt.ylabel("N")
    plt.show()
