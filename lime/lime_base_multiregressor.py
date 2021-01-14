"""
Custom modification of lime_base - that uses regression tree as default regressor.
"""
import numpy as np

from lime.lime_base_mod import LimeBaseMod
from sklearn.tree import DecisionTreeRegressor


class LimeBaseMultiRegressionTree(LimeBaseMod):
    """
    Class for learning a local surrogate model from perturbed data.
    Custom modification - uses regression tree as default regressor.
    """
    def __init__(self,
                 kernel_fn=None,
                 verbose=False,
                 random_state=None,
                 **regression_tree_kwargs):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            regression_tree_kwargs: additional keyword arguments to be passed to DecisionTreeRegressor
        """
        super().__init__(
            kernel_fn=kernel_fn,
            verbose=verbose,
            random_state=random_state
        )
        if len({"random_state", "max_depth"} & regression_tree_kwargs.keys()) > 0:
            raise RuntimeError("Argument in regression_tree_kwargs not allowed!")
        self.regression_tree_kwargs = regression_tree_kwargs

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='none',
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
            feature_selection: deprecated - it cedes responsibility to the Tree, not feature_selection.
            model_regressor: deprecated - DecisionTreeRegressor is always selected

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
                "none",
                label,
                DecisionTreeRegressor(
                    random_state=self.random_state,
                    max_depth=num_features,
                    **self.regression_tree_kwargs),
                neighborhood_data,
                neighborhood_labels,
                num_features)

        explanation = self._get_explanation(local_surrogate, used_features)

        return (None,   # deprecated field
                explanation,
                local_surrogate,
                used_features,
                weights)

    @staticmethod
    def _get_explanation(local_surrogate, used_features):

        explanation = sorted(
            zip(used_features, local_surrogate.feature_importances_),
            key=lambda x: np.abs(x[1]),
            reverse=True)

        return explanation

