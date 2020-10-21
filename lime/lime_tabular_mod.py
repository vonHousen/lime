"""
Functions for explaining classifiers that use tabular data (matrices) - custom modification TODO.
"""
import collections
import copy
from functools import partial
import json
import warnings

import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from pyDOE2 import lhs
from scipy.stats.distributions import norm

from lime.discretize import QuartileDiscretizer
from lime.discretize import DecileDiscretizer
from lime.discretize import EntropyDiscretizer
from lime.discretize import BaseDiscretizer
from lime.discretize import StatsDiscretizer
from . import explanation_mod
from . import lime_base
from lime.lime_tabular import LimeTabularExplainer


class LimeTabularExplainerMod(LimeTabularExplainer):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True
                and data is not sparse. Options are 'quartile', 'decile',
                'entropy' or a BaseDiscretizer instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            training_data_stats: a dict object having the details of training data
                statistics. If None, training data information will be used, only matters
                if discretize_continuous is True. Must have the following keys:
                means", "mins", "maxs", "stds", "feature_values",
                "feature_frequencies"
        """
        super().__init__(
            training_data,
            mode,
            training_labels,
            feature_names,
            categorical_features,
            categorical_names,
            kernel_width,
            kernel,
            verbose,
            class_names,
            feature_selection,
            discretize_continuous,
            discretizer,
            sample_around_instance,
            random_state,
            training_data_stats
        )

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         sampling_method='gaussian'):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        distances, domain_mapper, max_y, min_y, predicted_value, scaled_data, yss = \
            super()._get_prerequisites_for_explaining(
                data_row,
                distance_metric,
                num_samples,
                predict_fn,
                sampling_method
            )

        new_explanation = self.__create_explanation(
            distances, domain_mapper, labels, model_regressor, num_features, scaled_data, top_labels, yss
        )

        return new_explanation

    def __create_explanation(self,
                             distances,
                             domain_mapper,
                             labels,
                             model_regressor,
                             num_features,
                             scaled_data,
                             top_labels,
                             yss):
        """
        Factory method for creating and evaluating new explanation.
        """
        new_explanation = explanation_mod.ExplanationMod(
            domain_mapper,
            mode=self.mode,
            class_names=self.class_names)

        if self.mode == "classification":
            new_explanation.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                new_explanation.top_labels = list(labels)
                new_explanation.top_labels.reverse()
        else:
            raise NotImplementedError("Only classification is implemented :(")

        for label in labels:
            (intercept, feature_with_coef, predict_proba, prediction) = self.base.explain_instance_with_data(
                scaled_data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

            new_explanation.intercept[label] = intercept
            new_explanation.local_exp[label] = feature_with_coef
            new_explanation.probabilities_for_surrogate_model[label] = predict_proba
            new_explanation.predictions_for_surrogate_model[label] = prediction

            # TODO deprecated fields
            new_explanation.score = predict_proba
            new_explanation.local_pred = prediction

        return new_explanation


