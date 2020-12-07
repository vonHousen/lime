"""
Custom modification of lime_base - that uses decision tree as single local surrogate.
"""
import numpy as np

from lime.lime_base_mod import LimeBaseMod
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
from lime.tools import convert_binary_output_to_decimal

class LimeBaseSingleDecisionTree(LimeBaseMod):
    """
    Class for learning a local surrogate model from perturbed data.
    Custom modification - uses decision tree as local surrogate.
    """
    def __init__(self,
                 kernel_fn=None,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        super().__init__(
            kernel_fn=kernel_fn,
            verbose=verbose,
            random_state=random_state
        )
        self.model_classifier = DecisionTreeClassifier(random_state=self.random_state)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label_indices_to_explain,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: deprecated

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            explanation is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        data_to_train_local_surrogate, local_surrogate, used_features, weights =\
            self._train_local_surrogate(
                distances,
                feature_selection,
                label_indices_to_explain,
                self.model_classifier,
                neighborhood_data,
                neighborhood_labels,
                num_features)

        explanation = self._get_explanation(local_surrogate, used_features)

        return (None,   # deprecated field
                explanation,
                local_surrogate,
                data_to_train_local_surrogate,
                weights)

    @staticmethod
    def _get_explanation(local_surrogate, used_features):
        explanation = sorted(
            zip(used_features, local_surrogate.feature_importances_),
            key=lambda x: np.abs(x[1]),
            reverse=True)
        return explanation

    def _train_local_surrogate(self,
                               distances,
                               feature_selection,
                               label_indices_to_explain,
                               local_surrogate,
                               neighborhood_data,
                               neighborhood_labels,
                               num_features):
        weights = self.kernel_fn(distances)

        # predicted labels are the labels with the greatest probability - simple majority is not required
        predicted_labels = np.argmax(neighborhood_labels, axis=1)
        prediction_results = np.zeros_like(neighborhood_labels, dtype="int32")
        prediction_results[np.arange(prediction_results.shape[0]), predicted_labels] = 1
        classification_labels_columns = prediction_results[:, label_indices_to_explain]
        regression_labels_columns = neighborhood_labels[:, label_indices_to_explain]

        used_features = self._get_best_features(
            regression_labels_columns, feature_selection, neighborhood_data, num_features, weights)
        data_to_train_local_surrogate = neighborhood_data[:, used_features]
        local_surrogate.fit(
            data_to_train_local_surrogate,
            convert_binary_output_to_decimal(classification_labels_columns),
            sample_weight=weights)
        return data_to_train_local_surrogate, local_surrogate, used_features, weights

    def _get_best_features(self,
                           regression_labels_columns,
                           feature_selection,
                           neighborhood_data,
                           num_features,
                           weights):
        """
        Single classifier uses data with all labels at once.
        The self.feature_selection() method takes only one label at once, so it is executed in a loop, then - the most
            popular labels will be selected.
        """
        counter_for_feature = defaultdict(int)
        for column in regression_labels_columns.T:
            used_features = self.feature_selection(
                neighborhood_data,
                column,
                weights,
                num_features,
                feature_selection)
            for feature in used_features:
                counter_for_feature[feature] += 1

        sorted_features = \
            [feature for feature, _ in sorted(counter_for_feature.items(),
                                              key=lambda item: item[1],
                                              reverse=True)]
        best_features = sorted_features[:num_features]
        return best_features

