"""
Explanation class, with visualization functions - custom modification TODO.
"""
from scipy.special import softmax
from lime.explanation import Explanation
from sklearn.metrics import log_loss


class ExplanationMod(Explanation):
    """Object returned by explainers."""

    def __init__(self,   # TODO pass additional surrogate models here
                 domain_mapper,
                 mode='classification',
                 class_names=None,
                 random_state=None):
        """

        Initializer.

        Args:
            domain_mapper: must inherit from DomainMapper class
            type: "classification" or "regression"
            class_names: list of class names (only used for classification)
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        super().__init__(
            domain_mapper,
            mode,
            class_names,
            random_state
        )
        self.probabilities_for_surrogate_model = {}
        self.predictions_for_surrogate_model = {}

    def get_probabilities_for_explained_model(self):
        return self.predict_proba

    def get_probabilities_for_surrogate_model(self, normalized=False):
        predict_probas = [predict_proba for predict_proba in self.probabilities_for_surrogate_model.values()]
        if normalized:
            return softmax(predict_probas)
        else:
            return predict_probas

    def get_accuracy(self):
        return log_loss(
            y_true=self.get_probabilities_for_explained_model(),
            y_pred=self.get_probabilities_for_surrogate_model(normalized=True)
        )

    # TODO get_accuracy
    # TODO get_faithfulness
