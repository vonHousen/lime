"""
Functions for explaining classifiers that use tabular data (matrices) - custom modification TODO.
"""
import warnings

import numpy as np
from . import explanation_mod, lime_base_mod
from lime.lime_tabular import LimeTabularExplainer, TableDomainMapper
from scipy.special import softmax
from sklearn import metrics


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
            "classification",
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
            training_data_stats,
            custom_lime_base_constructor=lime_base_mod.LimeBaseMod
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
            self._get_prerequisites_for_explaining(
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

    def _get_prerequisites_for_explaining(self,
                                          data_row,
                                          distance_metric,
                                          num_samples,
                                          predict_fn,
                                          sampling_method):
        """
        Overridden.
        Method calculates prerequisites for explain_instance_with_data().
        """
        data_row, distances, inversed_data, scaled_data = self._process_raw_data(
            data_row,
            distance_metric,
            num_samples,
            sampling_method)

        yss = predict_fn(inversed_data)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if len(yss.shape) == 1:
            raise NotImplementedError()     # TODO implement for decision tree
        elif len(yss.shape) == 2:
            if self.class_names is None:
                self.class_names = [str(x) for x in range(yss[0].shape[0])]
            else:
                self.class_names = list(self.class_names)
            if not np.allclose(yss.sum(axis=1), 1.0):
                warnings.warn("""
                Prediction probabilties do not sum to 1, and
                thus does not constitute a probability space.
                Check that you classifier outputs probabilities
                (Not log probabilities, or actual class predictions).
                """)
        else:
            raise ValueError("Your model outputs "
                             "arrays with {} dimensions".format(len(yss.shape)))

        predicted_value = None
        min_y = None
        max_y = None

        categorical_features, discretized_feature_names, feature_indexes, feature_names, values = \
            self._process_features(data_row, scaled_data)

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)

        return distances, domain_mapper, max_y, min_y, predicted_value, scaled_data, yss

    @staticmethod
    def __evaluate_single_explainer(local_surrogate,
                                    training_data,
                                    labels,
                                    weights):
        """
        Evaluates single local surrogate on data used for its training (subset of features only),
        using built-in score function. Method uses coefficient of determination R^2 of the prediction as loss function.
        """
        prediction_score_on_training_data = local_surrogate.score(
            training_data,
            labels,
            sample_weight=weights)
        predictions_on_training_data = local_surrogate.predict(training_data)
        prediction_loss_on_training_data = metrics.mean_squared_error(
            y_true=labels, y_pred=predictions_on_training_data
        )
        prediction_on_explained_instance = local_surrogate.predict(
            training_data[0, :].reshape(1, -1))[0]
        return prediction_on_explained_instance, prediction_score_on_training_data, prediction_loss_on_training_data

    @staticmethod
    def __evaluate_ensemble(local_surrogates_ensemble,
                            data_subset_for_each_explainer,
                            training_data,
                            expected_probabilities):
        """
        Evaluates ensemble of local surrogates, by comparing softmax of their probabilities and expected ones,
        without weighing them. Note that each explainer might use different subset of data (different features).
        Loss function is MSE.
        """
        predicted_probabilities = np.zeros((training_data.shape[0], expected_probabilities.shape[1]), dtype="float")
        for label in range(expected_probabilities.shape[1]):
            local_surrogate = local_surrogates_ensemble[label]
            training_data_for_local_surrogate = data_subset_for_each_explainer[label]
            predicted_probabilities[:, label] = local_surrogate.predict(training_data_for_local_surrogate)
        predicted_probabilities_softmax = softmax(predicted_probabilities, axis=1)
        prediction_loss_on_training_data = metrics.mean_squared_error(
            y_true=expected_probabilities, y_pred=predicted_probabilities_softmax)
        squared_errors_matrix = (expected_probabilities - predicted_probabilities_softmax)**2

        return prediction_loss_on_training_data, squared_errors_matrix

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
            class_names=self.class_names)

        new_explanation.predict_proba = yss[0]
        if top_labels:
            label_indices_to_explain = np.argsort(yss[0])[-top_labels:]
            # legacy fields
            new_explanation.top_labels = list(label_indices_to_explain)
            new_explanation.top_labels.reverse()
        else:
            label_indices_to_explain = labels

        new_explanation.explained_labels_id = list(label_indices_to_explain)

        local_surrogates_ensemble = {}
        datasets_for_each_explainer = {}

        for label_idx in label_indices_to_explain:
            (intercept, feature_with_coef, local_surrogate, data_used_to_train_local_surrogate, examples_weights) =\
                self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label_idx,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

            local_surrogates_ensemble[label_idx] = local_surrogate
            datasets_for_each_explainer[label_idx] = data_used_to_train_local_surrogate

            labels_column = yss[:, label_idx]
            (prediction_on_explained_instance,
             prediction_score_on_training_data,
             prediction_loss_on_training_data) = self.__evaluate_single_explainer(
                local_surrogate,
                data_used_to_train_local_surrogate,
                labels_column,
                examples_weights)

            new_explanation.intercept[label_idx] = intercept
            new_explanation.local_exp[label_idx] = feature_with_coef
            new_explanation.probabilities_for_surrogate_model[label_idx] = prediction_on_explained_instance
            new_explanation.scores_on_generated_data[label_idx] = prediction_score_on_training_data
            new_explanation.losses_on_generated_data[label_idx] = prediction_loss_on_training_data

            # TODO deprecated fields
            new_explanation.score = prediction_score_on_training_data
            new_explanation.local_pred = prediction_on_explained_instance

        new_explanation.training_data_distances = distances
        (new_explanation.prediction_loss_on_training_data,
         new_explanation.squared_errors_matrix) = self.__evaluate_ensemble(
            local_surrogates_ensemble,
            data_subset_for_each_explainer=datasets_for_each_explainer,
            training_data=scaled_data,
            expected_probabilities=yss)

        return new_explanation


