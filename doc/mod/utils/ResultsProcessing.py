
import numpy as np
import matplotlib.pyplot as plt
from math import floor


class ResultsProcessing:

    def __init__(self,
                 models,
                 labels_count,
                 scores_for_surrogate_model,
                 losses_for_surrogate_model,
                 fidelity_loss_on_explanation,
                 fidelity_loss_on_generated_data,
                 fidelity_loss_distribution_quantiles):
        self.models = models
        self.labels_count = labels_count
        self.scores_for_surrogate_model = scores_for_surrogate_model
        self.losses_for_surrogate_model = losses_for_surrogate_model
        self.fidelity_loss_on_explanation = fidelity_loss_on_explanation
        self.fidelity_loss_on_generated_data = fidelity_loss_on_generated_data
        self.fidelity_loss_distribution = fidelity_loss_distribution_quantiles

    @classmethod
    def from_file(cls,
                  filename,
                  models,
                  labels_count):
        with open(f"{filename}.npy", "rb") as file:
            scores_for_surrogate_model = np.load(file)
            losses_for_surrogate_model = np.load(file)
            fidelity_loss_on_explanation = np.load(file)
            fidelity_loss_on_generated_data = np.load(file)
            fidelity_loss_distribution = np.load(file)
        return cls(models,
                   labels_count,
                   scores_for_surrogate_model,
                   losses_for_surrogate_model,
                   fidelity_loss_on_explanation,
                   fidelity_loss_on_generated_data,
                   fidelity_loss_distribution)

    def save_results(self,
                     filename):
        with open(f"{filename}.npy", "wb") as file:
            np.save(file, self.scores_for_surrogate_model)
            np.save(file, self.losses_for_surrogate_model)
            np.save(file, self.fidelity_loss_on_explanation)
            np.save(file, self.fidelity_loss_on_generated_data)
            np.save(file, self.fidelity_loss_distribution)

    def _plot_for_each_label(self,
                             data,
                             data_desc):
        for model_idx, (classifier_name, model) in enumerate(self.models):
            fig, axs = plt.subplots(nrows=1, ncols=self.labels_count)
            fig.set_figwidth(5 * self.labels_count)
            fig.set_figheight(4)
            fig.suptitle(
                f"Histogram for {data_desc} for surrogate model \n explained model: {classifier_name}", fontsize=16)
            fig.tight_layout(pad=2.0)
            for idx in range(self.labels_count):
                data_to_plot = data[model_idx, :, idx]
                mean_value = float(np.mean(data_to_plot))
                axs[idx].hist(data_to_plot, bins=30)
                axs[idx].axvline(mean_value, color="red", linestyle="--")
                axs[idx].set_title(f"#{idx} label {data_desc} | mean={round(mean_value, 4)}")
            plt.show()

    def plot_scores_for_surrogate_model(self):
        self._plot_for_each_label(
            data=self.scores_for_surrogate_model,
            data_desc="scores")

    def plot_losses_for_surrogate_model(self):
        self._plot_for_each_label(
            data=self.losses_for_surrogate_model,
            data_desc="losses")

    def _plot_for_each_model(self,
                             data,
                             data_desc):
        fig, axs = plt.subplots(nrows=len(self.models) // 3, ncols=3)
        fig.set_figwidth(15)
        fig.set_figheight(7)
        fig.suptitle(f"Histogram for {data_desc}", fontsize=16)
        fig.tight_layout(pad=2.0)
        for model_idx, (classifier_name, model) in enumerate(self.models):
            idx_row = floor(model_idx / 3)
            idx_col = model_idx % 3
            data_to_plot = data[model_idx, :]
            mean_value = float(np.mean(data_to_plot))
            axs[idx_row][idx_col].hist(data_to_plot, bins=30)
            axs[idx_row][idx_col].axvline(mean_value, color="red", linestyle="--")
            axs[idx_row][idx_col].set_title(
                f"explained model: {classifier_name} | mean={round(mean_value, 4)}")
        plt.show()

    def plot_fidelity_loss_on_explanation(self):
        self._plot_for_each_model(
            data=self.fidelity_loss_on_explanation,
            data_desc="fidelity losses on explanation")

    def plot_fidelity_losses_on_generated_data(self):
        self._plot_for_each_model(
            data=self.fidelity_loss_on_generated_data,
            data_desc="fidelity losses on generated data")

    def plot_fidelity_loss_distribution(self, domain_unit="Quantiles"):
        fig, axs = plt.subplots(nrows=len(self.models) // 3, ncols=3)
        fig.set_figwidth(15)
        fig.set_figheight(7)
        fig.suptitle(f"Mean distribution of fidelity loss on quantified distance", fontsize=16)
        fig.tight_layout(pad=4.0)
        for model_idx, (classifier_name, model) in enumerate(self.models):
            idx_row = floor(model_idx / 3)
            idx_col = model_idx % 3
            axs[idx_row][idx_col].plot(np.mean(self.fidelity_loss_distribution[model_idx, :], axis=0))
            axs[idx_row][idx_col].set_ylabel(f"Fidelity loss of samples")
            axs[idx_row][idx_col].set_xlabel(f"{domain_unit} of samples' distance")
            axs[idx_row][idx_col].set_title(f"explained model: {classifier_name}")
        plt.show()
