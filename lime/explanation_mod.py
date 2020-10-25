
from scipy.special import softmax
from lime.explanation import Explanation
from sklearn.metrics import log_loss


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
        self.probabilities_for_surrogate_model = {}
        self.predictions_for_surrogate_model = {}

    def get_probabilities_for_explained_model(self):
        """
        Returns prediction probabilities by explained model on given instance.
        """
        return self.predict_proba

    def get_probabilities_for_surrogate_model(self, normalized=False):
        """
        Returns prediction probabilities by surrogate model on given instance.
        """
        labels = list(self.probabilities_for_surrogate_model.keys())
        labels.sort()
        prediction_probabilities = []
        for label in labels:
            prediction_probabilities.append(self.probabilities_for_surrogate_model[label])
        if normalized:
            return softmax(prediction_probabilities)
        else:
            return prediction_probabilities

    def get_fidelity(self):
        """
        Function assesses efficiency of surrogate model by comparing its prediction probabilities and explained
        model's ones. Uses cross-entropy function for calculations.
        """
        expected = self.get_probabilities_for_explained_model()
        predicted = self.get_probabilities_for_surrogate_model(normalized=True)
        return log_loss(
            y_true=expected,
            y_pred=predicted
        )
