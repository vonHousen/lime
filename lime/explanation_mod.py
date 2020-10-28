
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
        self.losses_on_generated_data = {}
        self.prediction_loss_on_training_data = None

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

    def get_prediction_for_explained_model(self):
        """
        Returns prediction probabilities by explained model on given instance.
        """
        return self.predict_proba

    def get_prediction_for_surrogate_model(self, normalized=False, order="default"):
        """
        Returns prediction probabilities by surrogate model on given instance. In fact, surrogate model consists of
        a few models, each for a single label. Their predictions are made based on a subset of
        (most important) features and can be normalized using softmax function.
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
        Returns prediction scores by surrogate model on its training dataset. Each label (and each sub-explainer) was
        evaluated separately.
        """
        prediction_scores = []
        for label in self._get_labels_ordered(order):
            prediction_scores.append(self.scores_on_generated_data[label])
        return prediction_scores

    def get_losses_for_surrogate_model(self, order="default"):
        """
        Returns prediction losses by surrogate model on its training dataset. Each label (and each sub-explainer) was
        evaluated separately.
        """
        prediction_losses = []
        for label in self._get_labels_ordered(order):
            prediction_losses.append(self.losses_on_generated_data[label])
        return prediction_losses

    def get_fidelity_loss_on_explanation(self):
        """
        Function assesses efficiency of surrogate model by comparing its prediction's probabilities and explained
        model's ones - prediction on explained data. Surrogate's prediction is first normalized.
        Uses MSE as loss function.
        """
        expected = self.get_prediction_for_explained_model()
        predicted = self.get_prediction_for_surrogate_model(normalized=True)
        return metrics.mean_squared_error(
            y_true=expected,
            y_pred=predicted
        )

    def get_fidelity_loss_on_generated_data(self):
        """
        Function assesses efficiency of surrogate model by comparing its predictions' probabilities and explained
        model's ones - predictions on training data. Ensemble of sub-explainers was treated as a single regressor.
        Uses MSE as loss function.
        """
        return self.prediction_loss_on_training_data
