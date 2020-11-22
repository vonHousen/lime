import numpy as np


def custom_normalize(vector, axis=1):
    """
    Normalization function for ensemble of classifiers/regressors.
    Replaces each negative values (predictions' probabilities) to 0.0, and makes sure that all probabilities
    sum up to 1.0.
    """
    non_negative = np.array(vector)
    non_negative[non_negative < 0.0] = 0.0
    if len(non_negative.shape) == 1:
        summed = np.sum(non_negative)
        if summed != 0:
            normalized = non_negative / summed
        else:
            normalized = np.ones_like(non_negative, dtype="float32") / non_negative.shape[0]
    elif len(non_negative.shape) == 2:
        summed = np.sum(non_negative, axis=axis).reshape(-1, 1)
        degenerated_rows = summed.ravel() < 0.001
        normalized = non_negative.copy()
        normalized[np.logical_not(degenerated_rows), :] /= summed[np.logical_not(degenerated_rows), :]
        normalized[degenerated_rows, :] = \
            np.ones_like(non_negative[degenerated_rows, :], dtype="float32") / non_negative.shape[1]
    else:
        raise NotImplementedError()
    return normalized
