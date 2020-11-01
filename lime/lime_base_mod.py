"""
Contains custom modification of lime_base - TODO
"""
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from lime.lime_base import LimeBase


class LimeBaseMod(LimeBase):
    """Class for learning a local surrogate model from perturbed data"""
    def __init__(self,
                 kernel_fn,
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

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
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
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            explanation is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(
            neighborhood_data,
            labels_column,
            weights,
            num_features,
            feature_selection)
        data_to_train_local_surrogate = neighborhood_data[:, used_features]

        if model_regressor is None:
            model_regressor = Ridge(
                alpha=1,
                fit_intercept=True,
                random_state=self.random_state)
        local_surrogate = model_regressor
        local_surrogate.fit(
            data_to_train_local_surrogate,
            labels_column,
            sample_weight=weights)

        explanation = sorted(
            zip(used_features, local_surrogate.coef_),
            key=lambda x: np.abs(x[1]),
            reverse=True)

        return (local_surrogate.intercept_,
                explanation,
                local_surrogate,
                data_to_train_local_surrogate,
                weights)
