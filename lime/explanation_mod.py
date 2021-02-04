
from lime.explanation import Explanation
from sklearn import metrics, tree
import lime.tools as lime_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydotplus
import collections


class ExplanationMod(Explanation):
    """Modified explanation class, providing efficiency assessing functions."""

    def __init__(self,
                 domain_mapper,
                 explained_sample,
                 class_names=None,
                 random_state=None,
                 feature_names=None,
                 num_features=None,
                 ):
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
        self._local_surrogates_ensemble = {}
        self.feature_names = feature_names
        self.explained_sample = explained_sample
        self.num_features = num_features

        self.colors_palette_10_positive = {
            0.9: "#006837",
            0.8: "#1a9850",
            0.7: "#66bd63",
            0.6: "#a6d96a",
            0.5: "#d9ef8b"
        }
        self.colors_palette_10_unknown = "#fdae61"
        self.colors_palette_10_negative = "#d73027"
        self.unknown_boundary_probability = None

    @property
    def local_surrogates_ensemble(self):
        return self._local_surrogates_ensemble

    @local_surrogates_ensemble.setter
    def local_surrogates_ensemble(self, x):
        self.unknown_boundary_probability = 1. / len(x)
        self._local_surrogates_ensemble = x

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

    def get_decision_rules_for_explanation(self):

        clf = self._get_winning_subexplainer()
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold

        explanation_txt = f"Sklasyfikowano jako '{self.get_predicted_label()}', ponieważ:\n"
        node_indicator = clf.decision_path(self.explained_sample)
        leaf_id = clf.apply(self.explained_sample)

        sample_id = 0
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        for i, node_id in enumerate(node_index):
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            # check if value of the split feature for sample 0 is below threshold
            if (self.explained_sample[0, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            if i != 0:
                explanation_txt += "\toraz\t"
            else:
                explanation_txt += "\t\t"
            explanation_txt += "cecha {feature_name} (o wartości {value}) była {inequality} {threshold}\n".format(
                feature_name=self.feature_names[feature[node_id]],
                value=self.explained_sample[0, feature[node_id]],
                inequality=threshold_sign,
                threshold=round(threshold[node_id], 3)
            )

        return explanation_txt

    def render_explanation_tree(self,
                                file_to_render="tree.png"):

        clf = self._get_winning_subexplainer()
        if not isinstance(clf, tree.DecisionTreeRegressor):
            raise NotImplementedError("Explanation Tree can be rendered only for LTEMultiRegressionTree.")

        dot_data = tree.export_graphviz(
            clf,
            feature_names=self.feature_names,
            label="none",
            out_file=None,
            filled=True,
            impurity=False,
            proportion=True,
            rounded=True,
            rotate=True,
        )
        graph = pydotplus.graph_from_dot_data(dot_data)

        self._edit_edges_labels(graph, clf.tree_)
        self._color_nodes(graph, 0, clf.tree_)
        self._color_decision_path(clf, graph)

        graph.write_png(file_to_render)

    def _edit_edges_labels(self, graph, tree):

        self._edit_first_edges_labels(graph)

        children_left = tree.children_left
        children_right = tree.children_right

        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            # If the left and right child of a node is not the same we have a split node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them

            current_node = graph.get_node(str(node_id))[0]
            left_child = children_left[node_id]
            right_child = children_right[node_id]

            if not is_split_node:
                continue

            stack.append((left_child, depth + 1))
            stack.append((right_child, depth + 1))

            if node_id != 0:
                self._edit_outcoming_edges_labels(
                    graph,
                    current_node,
                    left_child,
                    right_child)

    @staticmethod
    def _edit_first_edges_labels(graph):
        # change first edges' labels
        for edge in graph.get_edge_list():
            label_angle = edge.obj_dict["attributes"].get("labelangle")
            if label_angle is not None:
                edge.set_labelangle(0.5 * int(label_angle))
                edge.set_labeldistance(5.0)

    @staticmethod
    def _edit_outcoming_edges_labels(graph,
                                     current_node,
                                     left_child,
                                     right_child,
                                     label_distance=2.5,
                                     label_angle_abs=30.,
                                     font_size=11):

        true_node = graph.get_node(str(left_child))[0]
        true_edge = graph.get_edge(current_node.get_name(), true_node.get_name())[0]
        true_edge.set_taillabel("True")
        true_edge.set_labeldistance(label_distance)
        true_edge.set_labelangle(label_angle_abs)
        true_edge.set_fontsize(font_size)

        false_node = graph.get_node(str(right_child))[0]
        false_edge = graph.get_edge(current_node.get_name(), false_node.get_name())[0]
        false_edge.set_taillabel("False")
        false_edge.set_labeldistance(label_distance)
        false_edge.set_labelangle(-1.0 * label_angle_abs)
        false_edge.set_fontsize(font_size)

    def _color_nodes(self,
                     graph,
                     node_id,
                     tree):
        values = tree.value
        children_left = tree.children_left
        children_right = tree.children_right
        is_split_node = children_left[node_id] != children_right[node_id]
        current_node = graph.get_node(str(node_id))[0]
        left_child = children_left[node_id]
        right_child = children_right[node_id]
        current_node_probability = values[node_id][0][0]

        if not is_split_node:
            selected_color = self._get_current_node_color(current_node_probability)

            current_node.set_fillcolor(selected_color)

        else:
            left_child_probability, left_child_color = \
                self._color_nodes(graph, left_child, tree)
            right_child_probability, right_child_color = \
                self._color_nodes(graph, right_child, tree)

            if left_child_color == right_child_color:
                selected_color = left_child_color
            elif left_child_probability > 0.5 and right_child_probability > 0.5:
                selected_color = self._get_current_node_color(current_node_probability)
            else:
                selected_color = "white"

            current_node.set_fillcolor(selected_color)

        self._edit_node_label(current_node, current_node_probability)
        return current_node_probability, selected_color

    def _edit_node_label(self,
                         node,
                         node_probability):
        label = node.obj_dict["attributes"].get("label")
        label_lines = label.split("\\n")
        is_a_leaf = len(label_lines) == 2
        if not is_a_leaf:
            edited_label = label_lines[0][1:]
        elif node_probability > 0.5:
            edited_label = self.get_predicted_label()
        elif node_probability < self.unknown_boundary_probability:
            edited_label = "Other label"
        else:
            edited_label = "Inconclusive"


        node.set_label(edited_label)

    def _color_decision_path(self, clf, graph):
        decision_path = clf.decision_path(self.explained_sample)
        for node_id, node_value in enumerate(decision_path.toarray()[0]):
            if node_value == 0:
                continue
            node = graph.get_node(str(node_id))[0]
            node.set_fillcolor('deepskyblue')

    def _get_current_node_color(self,
                                current_node_probability):
        if current_node_probability < self.unknown_boundary_probability:
            selected_color = self.colors_palette_10_negative
        elif current_node_probability < .50:
            selected_color = self.colors_palette_10_unknown
        else:
            for boundary_probability, color_code in self.colors_palette_10_positive.items():
                if current_node_probability > boundary_probability:
                    selected_color = color_code
                    break
            else:
                selected_color = "white"

        return selected_color

    def _get_winning_subexplainer(self):
        subexplainers_ensemble = self.local_surrogates_ensemble
        if len(subexplainers_ensemble) == 0:
            return None
        if len(subexplainers_ensemble) == 1:
            return subexplainers_ensemble[-1]

        prediction = self.get_prediction_for_explained_model()
        predicted_label_id = np.argmax(prediction)
        return subexplainers_ensemble[predicted_label_id]

    def get_predicted_label(self):
        prediction = self.get_prediction_for_explained_model()
        predicted_label_id = np.argmax(prediction)
        return self.class_names[predicted_label_id]

    def as_list(self, label=1, **kwargs):
        """Returns the explanation as a list.

        Args:
            label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
                Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper

        Returns:
            list of tuples (representation, weight), where representation is
            given by domain_mapper. Weight is a float.
        """
        label_to_use = label if self.mode == "classification" else self.dummy_label
        mapped_features_with_importance = self.domain_mapper.map_exp_ids(self.local_exp[label_to_use], **kwargs)
        if self.num_features is None:
            filtered_features_imporatance =\
                [(x[0], float(x[1])) for x in mapped_features_with_importance if float(x[1]) > 0.0]
        else:
            filtered_features_imporatance = []
            for i, (feature, feature_importance_str) in enumerate(mapped_features_with_importance):
                if i >= self.num_features:
                    break
                feature_importance = float(feature_importance_str)
                filtered_features_imporatance.append((feature, feature_importance))

        return filtered_features_imporatance

