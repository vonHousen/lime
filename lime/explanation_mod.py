
from lime.explanation import Explanation
from sklearn import metrics
import lime.tools as lime_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ExplanationMod(Explanation):
    """Modified explanation class, providing efficiency assessing functions."""

    def __init__(self,
                 domain_mapper,
                 class_names=None,
                 random_state=None):
        """
        Args:
            domain_mapper: must inherit from DomainMapper class
            class_names: list of class names (only used for classification)
            random_state: an integer or numpy.RandomState that will be used to generate random numbers.
                If None, the random state will be initialized using the internal numpy seed.
        """
        super().__init__(
            domain_mapper=domain_mapper,
            mode="classification",
            class_names=class_names,
            random_state=random_state
        )
        self.explained_labels_id = None
        self.prediction_for_explained_model = None
        self.prediction_for_surrogate_model = {}
        self.scores_on_generated_data = {}
        self.losses_on_generated_data = {}
        self.prediction_loss_on_training_data = None
        self.squared_errors_matrix = None
        self.training_data_distances = None
        self.cv_evaluation_results = {}
        self.ensemble_mse_for_cv = None
        self.local_surrogates_ensemble = {}

    def _get_labels_ordered(self, order):
        """

        Args:
             order:
                - ordered - the same order as used during creating explanation
                - default - labels sorted in ascending order, the default sequence as in original dataset

        Returns:
            list of labels in selected order.
        """
        if order == "ordered":
            return self.explained_labels_id
        elif order == "default":
            ordered_labels = list(self.explained_labels_id)
            ordered_labels.sort()
            return ordered_labels
        else:
            raise NotImplementedError()

    def get_prediction_for_explained_model(self):
        """
        Returns:
            list of predicted by explained model probabilities for each label.
        """
        return list(self.prediction_for_explained_model)

    def get_prediction_for_surrogate_model(self,
                                           normalized=False,
                                           order="default"):
        """
        Returns prediction probabilities by surrogate model on given instance. In fact, surrogate model consists of
        a few models, each for a single label. Their predictions are made based on a subset of
        (most important) features.

        Args:
            normalized - if True, predictions are normalized using lime.utils.custom_normalize function.
            order - order of returned labels' probabilities - see _get_labels_ordered().

        Returns:
            list of predicted by local surrogate probabilities for each label.
        """
        prediction_probabilities = []
        for label in self._get_labels_ordered(order):
            prediction_probabilities.append(self.prediction_for_surrogate_model[label])
        if normalized:
            return list(lime_utils.custom_normalize(prediction_probabilities))
        else:
            return prediction_probabilities

    def get_scores_for_surrogate_model(self,
                                       order="default"):
        """
        Returns prediction scores by surrogate model on its training dataset. Each label (and each sub-explainer) was
        evaluated separately.

        Args:
            order - order of returned scores - see _get_labels_ordered().

        Returns:
            list of R2 scores for each sub-explainer.
        """
        prediction_scores = []
        for label in self._get_labels_ordered(order):
            prediction_scores.append(self.scores_on_generated_data[label])
        return prediction_scores

    def get_losses_for_surrogate_model(self,
                                       order="default"):
        """
        Returns prediction losses of surrogate models on their training datasets. Each label (and each sub-explainer)
        was evaluated separately.
        Uses MSE as loss function.

        Args:
            order - order of returned losses - see _get_labels_ordered().

        Returns:
            list of MSE for each sub-explainer.
        """
        prediction_losses = []
        for label in self._get_labels_ordered(order):
            prediction_losses.append(self.losses_on_generated_data[label])
        return prediction_losses

    def get_losses_for_cv_model(self,
                                order="default",
                                out="raw"):
        """
        Returns prediction losses of surrogate models evaluated with KFold. Each label (and each sub-explainer) was
        evaluated separately.
        Uses MSE as loss function.

        Args:
            order - order of returned losses - see _get_labels_ordered().

        Returns:
            array of MSE for each sub-explainer, for each validation.
        """
        evaluation_results = []
        for label in self._get_labels_ordered(order):
            evaluation_results.append(self.cv_evaluation_results[label])

        raw_results = np.array(evaluation_results)
        if out == "raw":
            return raw_results
        elif out == "mean":
            return np.mean(raw_results, axis=1)
        elif out == "std":
            return np.std(raw_results, axis=1)
        else:
            raise NotImplementedError()

    def get_fidelity_loss_on_explanation(self):
        """
        Function assesses efficiency of surrogate model by comparing its prediction's probabilities and explained
        model's ones - prediction on explained data. Surrogate's prediction is first normalized.
        Uses MSE as loss function.

        Returns:
            scalar - fidelity loss (MSE) calculated on data instance to be explained.
        """
        expected = self.get_prediction_for_explained_model()
        predicted = self.get_prediction_for_surrogate_model(normalized=True, order="default")
        return metrics.mean_squared_error(
            y_true=expected,
            y_pred=predicted
        )

    def get_fidelity_loss_on_generated_data(self):
        """
        Function assesses efficiency of surrogate model by comparing its predictions' probabilities and explained
        model's ones - predictions on training data. Ensemble of sub-explainers was treated as a single regressor.
        Uses MSE as loss function.

        Returns:
            scalar - fidelity loss (MSE) calculated on complete generated unweighted dataset.
        """
        return self.prediction_loss_on_training_data

    def get_fidelity_loss_on_kfold(self, out="raw"):
        """
        Function assesses efficiency of surrogate model by comparing its predictions' probabilities and explained
        model's ones - predictions on test data (KFold). Ensemble of sub-explainers was treated as a single regressor.
        Uses MSE as loss function.

        Returns:
            list - fidelity loss (MSE) calculated on complete generated unweighted dataset.
        """
        raw_results = np.array(self.ensemble_mse_for_cv)
        if out == "raw":
            return raw_results
        elif out == "mean":
            return np.mean(raw_results)
        elif out == "std":
            return np.std(raw_results)
        else:
            raise NotImplementedError()

    def get_fidelity_loss_distribution(self, bins=None, quantiles=None):
        """
        Functions returns fidelity loss of surrogate model based on every generated data instance used for its training.

        Args:
            bins: count of bins to use for grouping. If not none, returned errors are grouped in bins of equal width
                (bins are created based on distance from original data instance).
            quantiles: count of quantiles to use for grouping. If not none, returned errors are grouped in quantiles
                (bins are created based on distance from original data instance).

        Returns:
            pd.Series of squared errors for each data instance.
        """
        training_data_squared_errors = np.mean(self.squared_errors_matrix, axis=1)
        training_data_distances = self.training_data_distances
        sorted_sequence = np.argsort(training_data_distances)
        training_data = pd.DataFrame(
            data=np.concatenate((
                training_data_squared_errors[sorted_sequence].reshape((-1, 1)),
                training_data_distances[sorted_sequence].reshape((-1, 1))),
                axis=1),
            columns=["squared_error", "distance"]
        )
        if bins is not None:
            training_data["distance_binned"] = pd.cut(
                training_data["distance"],
                bins=bins,
                right=False,
                labels=[bin_id / bins for bin_id in range(1, bins + 1)])
            training_data = training_data[["squared_error", "distance_binned"]]\
                .groupby("distance_binned").mean()
        elif quantiles is not None:
            training_data["distance_quantile"] = pd.qcut(
                training_data["distance"],
                q=quantiles,
                labels=False,
                duplicates="drop")
            training_data = training_data[["squared_error", "distance_quantile"]]\
                .groupby("distance_quantile").mean()
            missing_rows = quantiles - len(training_data.index)
            if missing_rows > 0:
                first_new_row = len(training_data.index)
                training_data = pd.concat([
                        training_data,
                        pd.Series(index=[i for i in range(first_new_row, first_new_row + missing_rows)])
                    ])
        else:
            training_data.set_index("distance", inplace=True)

        return training_data["squared_error"]

    def plot_fidelity_map(self):
        squared_errors = self.get_fidelity_loss_distribution()

        plt.scatter(
            x=(squared_errors.index) / np.mean(squared_errors.index.values),
            y=squared_errors,
            label="generated data instance",
            marker=".")
        plt.scatter(
            x=(squared_errors.index[0]) / np.mean(squared_errors.index.values),
            y=squared_errors.iloc[0],
            label="original data instance",
            marker=".")
        plt.xlabel("normalized distance from original instance")
        plt.ylabel("squared error of data instance")
        plt.legend()
        plt.show()
