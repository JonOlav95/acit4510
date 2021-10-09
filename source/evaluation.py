import sklearn
from matplotlib import pyplot as plt


def precision_recall_values(y_predict, y_new, threshold):

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for i in range(len(y_predict)):
        print("PRED: " + str(y_predict[i][1]) + "\tREAL: " + str(y_new.iloc[i][0]))

        if y_predict[i][1] >= threshold:

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


def precision_recall_plot(y_predict, y_new):
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
