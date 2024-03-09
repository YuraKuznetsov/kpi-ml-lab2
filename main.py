import math
import random

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc


def analyze_data_frame(df, thresholds_step_percentage):
    thresholds = [i / 100 for i in range(100, -1, -thresholds_step_percentage)]

    for model_name in ["Model_1", "Model_2"]:
        metrics = {
            "accuracy": [], "precision": [], "recall": [],
            "F_scores": [], "MCC": [], "BA": [], "J": [],
            "TPR": [], "FPR": []
        }

        print("\n", model_name)
        for i, threshold in enumerate(thresholds):
            calculate_metrics_values(metrics, df, model_name, threshold)

            print("\nThreshold", threshold)
            for key in metrics:
                print(key, "-", metrics[key][i])

        showPRC(metrics["recall"], metrics["precision"], model_name)
        showROC(metrics["FPR"], metrics["TPR"], model_name)
        show_metrics_for_thresholds(metrics, thresholds, model_name)


def calculate_metrics_values(metrics, df, model_name, t):
    TP = df.loc[(df["GT"] == 1) & (df[model_name] > t)].shape[0]
    FP = df.loc[(df["GT"] == 0) & (df[model_name] > t)].shape[0]
    TN = df.loc[(df["GT"] == 0) & (df[model_name] <= t)].shape[0]
    FN = df.loc[(df["GT"] == 1) & (df[model_name] <= t)].shape[0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP != 0 else 0
    recall = TP / (TP + FN) if TP != 0 else 0

    B = 1
    F_scores = (1 + B**2) * precision * recall / (B**2 * (precision + recall)) if precision != 0 or recall != 0 else 0

    try:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except ZeroDivisionError:
        MCC = None

    BA = 0.5 * (TP / (TP + FN) + TN / (TN + FP))
    J = TP / (TP + FN) + TN / (TN + FP) - 1

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    metrics["accuracy"].append(accuracy)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["F_scores"].append(F_scores)
    metrics["MCC"].append(MCC)
    metrics["BA"].append(BA)
    metrics["J"].append(J)
    metrics["TPR"].append(TPR)
    metrics["FPR"].append(FPR)


def show_metrics_for_thresholds(metrics, thresholds, model_name):
    metrics.pop("TPR", None)
    metrics.pop("FPR", None)
    metrics["MCC"] = list(map(lambda x: 0 if x is None else x, metrics["MCC"]))

    plt.figure(figsize=(10, 6))
    for metric, values in metrics.items():
        plt.plot(thresholds, values, label=metric)

    # Marking maximum value for each metric
    for metric, values in metrics.items():
        max_value = max(values)
        max_index = values.index(max_value)
        plt.scatter(thresholds[max_index], max_value, marker='o', color='red')
        plt.text(thresholds[max_index], max_value, f"({thresholds[max_index]}, {round(max_value, 3)})", fontsize=9)

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Metrics for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()


def showPRC(recall, precision, model_name):
    AUC_PRC = auc(recall, precision)
    plt.figure()

    # Plot the ROC curve with a label displaying the ROC AUC score
    plt.plot(recall, precision, marker="o", color='darkorange', lw=2,
             label='PRC curve (area = %0.2f)' % AUC_PRC)

    # Plot a dashed diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set the x and y-axis limits
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    # Label the x and y-axes
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Set the title of the plot
    plt.title(f'Precision-Recall Curve ({model_name})')

    # Add a legend to the plot
    plt.legend(loc='lower right')

    # Display the ROC curve plot
    plt.show()


def showROC(FPR, TPR, model_name):
    AUC_PRC = auc(FPR, TPR)
    plt.figure()

    # Plot the ROC curve with a label displaying the ROC AUC score
    plt.plot(FPR, TPR, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % AUC_PRC)

    # Plot a dashed diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set the x and y-axis limits
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    # Label the x and y-axes
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Set the title of the plot
    plt.title(f'Receiver Operating Characteristic ({model_name})')

    # Add a legend to the plot
    plt.legend(loc='lower right')

    # Display the ROC curve plot
    plt.show()


def create_new_df(df):
    birth_date = "09-04"
    month = int(birth_date.split("-")[1])
    k = month % 4
    percentage_to_delete = 50 + 10 * k
    print("Percentage of deleted objects:", percentage_to_delete)

    class_1_rows = df[df['GT'] == 1]
    num_rows_to_delete = int(len(class_1_rows) * (percentage_to_delete / 100))
    rows_to_delete = random.sample(list(class_1_rows.index), num_rows_to_delete)

    return df.drop(rows_to_delete)


def main():
    # 1
    df = pd.read_csv("KM-12-1.csv")

    # 2
    print(df.groupby("GT")["GT"].count().to_string())

    # 3
    analyze_data_frame(df, 10)

    # 5
    df = create_new_df(df)

    # 6
    print(df.groupby("GT")["GT"].count().to_string())

    # 7
    # analyze_data_frame(df, 10)


if __name__ == '__main__':
    main()
