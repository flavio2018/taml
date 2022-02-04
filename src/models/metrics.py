from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from seaborn import heatmap
import matplotlib.pyplot as plt


def evaluate_preds(predictions, targets):
    print("MAE:", mean_absolute_error(predictions.T, targets.T))
    print("MSE:", mean_squared_error(predictions.T, targets.T))


def plot_confusion_matrix(predictions, targets, filename):
    discretized_predictions = [round(p) for p in predictions.flatten()]
    matrix = confusion_matrix(targets.T.squeeze(), discretized_predictions)
    print(matrix)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = heatmap(matrix, ax=ax)
    plt.savefig(f"../reports/figures/{filename}.pdf", format="pdf")
