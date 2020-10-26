
from scipy.special import softmax
from lime.explanation import Explanation
from sklearn import metrics


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
        self.used_labels = None
        self.probabilities_for_surrogate_model = {}
        self.scores_on_generated_data = {}
        self.prediction_training_loss = None

    def _get_labels_ordered(self, order):
        """
        Returns labels used in explanation in selected order.
        :param order:
                - default - order used in explaining
                - ordered - labels sorted in ascending order
        """
        if order == "default":
            return self.used_labels
        elif order == "ordered":
            ordered_labels = list(self.used_labels)
            ordered_labels.sort()
            return ordered_labels
        else:
            raise NotImplementedError()

    def get_probabilities_for_explained_model(self):
        """
        Returns prediction probabilities by explained model on given instance.
        """
        return self.predict_proba

    def get_probabilities_for_surrogate_model(self, normalized=False, order="default"):
        """
        Returns prediction probabilities by surrogate model on given instance.
        """
        prediction_probabilities = []
        for label in self._get_labels_ordered(order):
            prediction_probabilities.append(self.probabilities_for_surrogate_model[label])
        if normalized:
            return softmax(prediction_probabilities)
        else:
            return prediction_probabilities

    def get_scores_for_surrogate_model(self, order="default"):
        """
        Returns prediction scores by surrogate model on its training dataset.
        """
        prediction_scores = []
        for label in self._get_labels_ordered(order):
            prediction_scores.append(self.scores_on_generated_data[label])
        return prediction_scores

    def get_explanation_fidelity_loss(self):
        """
        Function assesses efficiency of surrogate model by comparing its prediction probabilities and explained
        model's ones. Uses MSE as loss function.
        """
        expected = self.get_probabilities_for_explained_model()
        predicted = self.get_probabilities_for_surrogate_model(normalized=True)
        return metrics.mean_squared_error(
            y_true=expected,
            y_pred=predicted
        )
