"""
Custom modification of LimeTabularExplainerMod - uses LimeBaseSingleDecisionTree as custom_lime_base.
"""
import warnings
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict

import lime.tools as lime_utils
import lime.explanation_mod as explanation_mod
from lime.lime_tabular_mod import LimeTabularExplainerMod
from lime.lime_base_singleclassifier import LimeBaseSingleDecisionTree
from lime.tools import convert_decimal_output_to_binary
from sklearn import metrics


class LTESingleDecisionTree(LimeTabularExplainerMod):
    """
    Custom modification of LimeTabularExplainerMod - uses LimeBaseSingleDecisionTree as custom_lime_base.
    """

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
                 training_data_stats=None,
                 with_kfold=None,
                 use_inversed_data_for_training=False,
                 **decision_tree_kwargs):
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
            custom_lime_base=LimeBaseSingleDecisionTree(**decision_tree_kwargs),
            with_kfold=with_kfold,
            use_inversed_data_for_training=use_inversed_data_for_training
        )

    def _create_explanation(self,
                            distances,
                            domain_mapper,
                            labels,
                            model_regressor,
                            num_features,
                            scaled_data,
                            top_labels,
                            yss,
                            inversed_data):
        """
        Factory method for creating and evaluating new explanation.
        """
        new_explanation = explanation_mod.ExplanationMod(
            domain_mapper,
            inversed_data[0].reshape(1, -1),
            class_names=self.class_names,
            feature_names=self.feature_names
        )

        label_indices_to_explain, prediction_results = \
            self._prepare_explanation(distances, new_explanation, top_labels, yss)

        if self.with_kfold is not None:
            kf = KFold(n_splits=self.with_kfold, shuffle=True, random_state=42)
        if self.use_inversed_data_for_training:
            data_to_train_local_surrogate = inversed_data
        else:
            data_to_train_local_surrogate = scaled_data

        (_,
         feature_with_coef,
         local_surrogate,
         used_features,
         examples_weights) = \
            self.base.explain_instance_with_data(
                data_to_train_local_surrogate,
                yss,
                distances,
                label_indices_to_explain,
                num_features,
                feature_selection=self.feature_selection)

        data_used_to_train_local_surrogate = data_to_train_local_surrogate[:, used_features]
        new_explanation.local_surrogates_ensemble[-1] = local_surrogate

        (new_explanation.prediction_for_surrogate_model,
         new_explanation.scores_on_generated_data,
         new_explanation.losses_on_generated_data,
         squared_errors_matrix_for_label_idx) = self._evaluate_explainer(
            local_surrogate,
            data_used_to_train_local_surrogate,
            prediction_results)

        for i, label_idx in enumerate(label_indices_to_explain):
            new_explanation.local_exp[label_idx] = feature_with_coef

        if self.with_kfold is not None:
            (cv_evaluation_results_for_label_idx,
             cv_subexplainers,
             cv_used_features) =\
                self._cross_validate_subexplainer(
                    distances,
                    label_indices_to_explain,
                    model_regressor,
                    num_features,
                    data_to_train_local_surrogate,
                    yss,
                    prediction_results,
                    kf)
            new_explanation.cv_evaluation_results = cv_evaluation_results_for_label_idx

        if top_labels == yss.shape[1]:
            (new_explanation.prediction_loss_on_training_data,
             new_explanation.squared_errors_matrix) = self._evaluate_ensemble(
                local_surrogate,
                label_indices_to_explain,
                data_subset_for_each_explainer=data_used_to_train_local_surrogate,
                training_data=data_to_train_local_surrogate,
                expected_probabilities=prediction_results
            )

            if self.with_kfold is not None:
                new_explanation.ensemble_mse_for_cv = self._cross_validate_ensemble(
                    kf,
                    label_indices_to_explain,
                    dataset=data_to_train_local_surrogate,
                    expected_probabilities=prediction_results,
                    cv_subexplainers=cv_subexplainers,
                    cv_used_features=cv_used_features
                )

        return new_explanation

    def _cross_validate_subexplainer(self,
                                     distances,
                                     label_indices_to_explain,
                                     model_regressor,
                                     num_features,
                                     training_data,
                                     yss,
                                     prediction_results,
                                     kf):
        cv_evaluation_results_for_label_idx = defaultdict(list)
        cv_subexplainers = []
        cv_used_features = []
        for train_indices, test_indices in kf.split(training_data):
            (_, _, cv_subexplainer, used_features, _) =\
                self.base.explain_instance_with_data(
                    training_data[train_indices],
                    yss[train_indices],
                    distances[train_indices],
                    label_indices_to_explain,
                    num_features,
                    feature_selection=self.feature_selection)

            row_x_indices = np.reshape(test_indices, (-1, 1))
            column_x_indices = np.repeat(np.reshape(used_features, (1, -1)), test_indices.shape[0], axis=0)
            test_x_data = training_data[row_x_indices, column_x_indices]
            test_y_data = prediction_results[test_indices]
            predicted = convert_decimal_output_to_binary(
                cv_subexplainer.predict(test_x_data), classes_count=test_y_data.shape[1])

            for label_idx in label_indices_to_explain:
                cv_evaluation_results_for_label_idx[label_idx].append(metrics.mean_squared_error(
                    test_y_data[:, label_idx], predicted[:, label_idx]
                ))
            cv_subexplainers.append(cv_subexplainer)
            cv_used_features.append(used_features)

        return (cv_evaluation_results_for_label_idx,
                cv_subexplainers,
                cv_used_features)

    @staticmethod
    def _evaluate_explainer(local_surrogate,
                            training_data,
                            prediction_results):
        """
        Evaluates the local surrogate on data used for its training (subset of features only),
        using built-in score function. Because local surrogate is a classifier, prediction score is not calculated.
        """
        prediction_on_explained_instance_for_label_idx = {}
        prediction_score_on_training_data_for_label_idx = {}
        prediction_loss_on_training_data_for_label_idx = {}
        squared_errors_matrix_for_label_idx = {}

        predictions_on_training_data_matrix = convert_decimal_output_to_binary(
            local_surrogate.predict(training_data), classes_count=prediction_results.shape[1])

        for label_idx in range(prediction_results.shape[1]):
            predictions_on_training_data = predictions_on_training_data_matrix[:, label_idx]
            expected_predictions = prediction_results[:, label_idx]
            squared_errors_matrix_for_label_idx[label_idx] = \
                (expected_predictions - predictions_on_training_data) ** 2
            prediction_loss_on_training_data_for_label_idx[label_idx] = metrics.mean_squared_error(
                y_true=expected_predictions, y_pred=predictions_on_training_data
            )
            prediction_on_explained_instance_for_label_idx[label_idx] = predictions_on_training_data[0]
            prediction_score_on_training_data_for_label_idx[label_idx] = 1.0

        return (prediction_on_explained_instance_for_label_idx,
                prediction_score_on_training_data_for_label_idx,
                prediction_loss_on_training_data_for_label_idx,
                squared_errors_matrix_for_label_idx)

    @staticmethod
    def _get_prediction_results(yss):
        # predicted labels are the labels with the greatest probability - simple majority is not required
        predicted_labels = np.argmax(yss, axis=1)
        prediction_results = np.zeros_like(yss, dtype="int32")
        prediction_results[np.arange(prediction_results.shape[0]), predicted_labels] = 1
        return prediction_results

    @staticmethod
    def _cross_validate_ensemble(kf,
                                 label_indices_to_explain,
                                 dataset,
                                 expected_probabilities,
                                 cv_subexplainers,
                                 cv_used_features):
        """
        Evaluates ensemble of subexplainers with KFold, by comparing their normalized probabilities and expected ones,
        without weighing them. Note that each explainer might use different subset of data (different features).
        Loss function is MSE, normalization function: lime.utils.custom_normalize.
        """
        evaluation_results = []
        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(dataset)):
            row_x_indices_for_fold = np.reshape(test_indices, (-1, 1))
            test_y_data_for_fold = expected_probabilities[test_indices]

            subexplainer = cv_subexplainers[fold_idx]
            used_features_for_subexplainer = cv_used_features[fold_idx]
            column_x_indices = np.repeat(
                np.reshape(used_features_for_subexplainer, (1, -1)),
                test_indices.shape[0],
                axis=0)
            test_x_data = dataset[row_x_indices_for_fold, column_x_indices]
            predicted_probabilities = convert_decimal_output_to_binary(
                subexplainer.predict(test_x_data), classes_count=len(label_indices_to_explain))

            predicted_probabilities_normalized = lime_utils.custom_normalize(predicted_probabilities, axis=1)
            mse_for_fold = metrics.mean_squared_error(
                y_true=test_y_data_for_fold, y_pred=predicted_probabilities_normalized)
            evaluation_results.append(mse_for_fold)

        return evaluation_results

    @staticmethod
    def _evaluate_ensemble(local_surrogate,
                           label_indices_to_explain,
                           data_subset_for_each_explainer,
                           training_data,
                           expected_probabilities):

        predicted_probabilities = convert_decimal_output_to_binary(
            local_surrogate.predict(training_data), classes_count=expected_probabilities.shape[1])
        prediction_loss_on_training_data = metrics.mean_squared_error(
            y_true=expected_probabilities, y_pred=predicted_probabilities)
        squared_errors_matrix = (expected_probabilities - predicted_probabilities)**2

        return prediction_loss_on_training_data, squared_errors_matrix
