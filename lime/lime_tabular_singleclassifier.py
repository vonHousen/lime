"""
Custom modification of LimeTabularExplainerMod - uses LimeBaseSingleDecisionTree as custom_lime_base.
"""
import warnings
import numpy as np
from sklearn.model_selection import KFold

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
                 max_depth=None,
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
            custom_lime_base=LimeBaseSingleDecisionTree(
                max_depth=max_depth,
                **decision_tree_kwargs
            ),
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
            class_names=self.class_names)

        label_indices_to_explain, prediction_results = \
            self._prepare_explanation(distances, new_explanation, top_labels, yss)
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
        (prediction_on_explained_instance,
         prediction_score_on_training_data,
         prediction_loss_on_training_data,
         squared_errors_matrix) = self._evaluate_explainer(
            local_surrogate,
            data_used_to_train_local_surrogate,
            prediction_results)

        if self.with_kfold is not None:
            kf = KFold(n_splits=self.with_kfold, shuffle=True, random_state=42)
            (cv_evaluation_results, cv_subexplainers, cv_used_features) = self._cross_validate_subexplainer(
                distances,
                label_indices_to_explain,
                model_regressor,
                num_features,
                data_to_train_local_surrogate,
                yss,
                prediction_results,
                kf)

        for i, label_idx in enumerate(label_indices_to_explain):

            if self.with_kfold is not None:
                new_explanation.cv_evaluation_results[label_idx] = cv_evaluation_results

            new_explanation.local_exp[label_idx] = feature_with_coef
            new_explanation.prediction_for_surrogate_model[label_idx] = prediction_on_explained_instance[i]
            new_explanation.scores_on_generated_data[label_idx] = prediction_score_on_training_data
            new_explanation.losses_on_generated_data[label_idx] = prediction_loss_on_training_data

        if top_labels == yss.shape[1]:
            new_explanation.prediction_loss_on_training_data = prediction_loss_on_training_data
            new_explanation.squared_errors_matrix = squared_errors_matrix

            if self.with_kfold is not None:
                new_explanation.ensemble_mse_for_cv = cv_evaluation_results


        return new_explanation

    def _cross_validate_subexplainer(self,
                                     distances,
                                     label_idx,
                                     model_regressor,
                                     num_features,
                                     training_data,
                                     yss,
                                     labels_column,
                                     kf):
        """
        Performs cross validation on given data.
        :return: np.array of evaluation results - MSE.
        """
        evaluation_results = []
        cv_subexplainers = []
        used_features_all = []
        for train_indices, test_indices in kf.split(training_data):
            (_, _, cv_subexplainer, used_features, _) =\
                self.base.explain_instance_with_data(
                    training_data[train_indices],
                    yss[train_indices],
                    distances[train_indices],
                    label_idx,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)
            row_x_indices = np.reshape(test_indices, (-1, 1))
            column_x_indices = np.repeat(np.reshape(used_features, (1, -1)), test_indices.shape[0], axis=0)
            test_x_data = training_data[row_x_indices, column_x_indices]
            test_y_data = labels_column[test_indices]
            predicted = convert_decimal_output_to_binary(
                cv_subexplainer.predict(test_x_data), classes_count=test_y_data.shape[1])
            evaluation_results.append(metrics.mean_squared_error(test_y_data, predicted))

            cv_subexplainers.append(cv_subexplainer)
            used_features_all.append(used_features)

        return evaluation_results, cv_subexplainers, used_features_all


    @staticmethod
    def _evaluate_explainer(local_surrogate,
                            training_data,
                            labels):
        """
        Evaluates the local surrogate on data used for its training (subset of features only),
        using built-in score function. Because local surrogate is a classifier, prediction score is not calculated.
        """
        prediction_score_on_training_data = 0.0
        predictions_on_training_data = convert_decimal_output_to_binary(
            local_surrogate.predict(training_data), classes_count=labels.shape[1])
        squared_errors_matrix = (labels - predictions_on_training_data)**2
        prediction_loss_on_training_data = metrics.mean_squared_error(
            y_true=labels, y_pred=predictions_on_training_data
        )
        prediction_on_explained_instance = predictions_on_training_data[0, :]

        return prediction_on_explained_instance, \
               prediction_score_on_training_data, \
               prediction_loss_on_training_data, \
               squared_errors_matrix

    @staticmethod
    def _get_prediction_results(yss):
        # predicted labels are the labels with the greatest probability - simple majority is not required
        predicted_labels = np.argmax(yss, axis=1)
        prediction_results = np.zeros_like(yss, dtype="int32")
        prediction_results[np.arange(prediction_results.shape[0]), predicted_labels] = 1
        return prediction_results