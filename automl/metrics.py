import numpy as np


def weighted_quantile_loss(quantile, y_true, quantile_pred):
    """
    The Weighted Quantile Loss (wQL) metric measures the accuracy of predictions
    at a specified quantile.

    :param quantile: Specific quantile to be analyzed.
    :param y_true: The observed values.
    :param quantile_pred: Quantile values that the model predicted.

    """

    max_vec = np.vectorize(max) # vectorize max function to apply over matrices

    first_term = quantile * max_vec(y_true - quantile_pred, 0)
    second_term = (1 - quantile) * max_vec(quantile_pred - y_true, 0)

    loss = 2 * (np.sum(first_term + second_term) / np.sum(y_true))

    return loss


def weighted_absolute_percentage_error(y_true, y_pred):
    """
    The Weighted Absolute Percentage Error (WAPE) metric measures the overall
    deviation of forecasted values from observed values.

    :param y_true: The observed values.
    :param y_pred: The predicted values.

    """

    absolute_error_sum = np.sum(np.abs(y_true - y_pred))

    loss = absolute_error_sum / np.sum(np.abs(y_true))

    return loss