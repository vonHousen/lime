
import numpy as np
import matplotlib.pyplot as plt
from math import floor

"""
Module containing classes used for results processing (postprocessing phase). 
"""


class ResultsProcessing:

    def __init__(self,
                 models,
                 labels_count,
                 scores_for_surrogate_model,
                 losses_for_surrogate_model,
                 losses_mean_for_cv_model,
                 losses_std_for_cv_model,
                 fidelity_loss_on_explanation,
                 fidelity_loss_on_generated_data,
                 fidelity_loss_on_kfold_mean,
                 fidelity_loss_on_kfold_std,
                 fidelity_loss_distribution_quantiles
                 ):
        self.models = models
        self.labels_count = labels_count
        self.scores_for_surrogate_model = scores_for_surrogate_model
        self.losses_for_surrogate_model = losses_for_surrogate_model
        self.losses_mean_for_cv_model = losses_mean_for_cv_model
        self.losses_std_for_cv_model = losses_std_for_cv_model
        self.fidelity_loss_on_explanation = fidelity_loss_on_explanation
        self.fidelity_loss_on_generated_data = fidelity_loss_on_generated_data
        self.fidelity_loss_on_kfold_mean = fidelity_loss_on_kfold_mean
        self.fidelity_loss_on_kfold_std = fidelity_loss_on_kfold_std
        self.fidelity_loss_distribution = fidelity_loss_distribution_quantiles

    @classmethod
    def from_file(cls,
                  filename,
                  models,
                  labels_count):
        with open(f"{filename}.npy", "rb") as file:
            scores_for_surrogate_model = np.load(file)
            losses_for_surrogate_model = np.load(file)
            losses_mean_for_cv_model = np.load(file)
            losses_std_for_cv_model = np.load(file)
            fidelity_loss_on_explanation = np.load(file)
            fidelity_loss_on_generated_data = np.load(file)
            fidelity_loss_on_kfold_mean = np.load(file)
            fidelity_loss_on_kfold_std = np.load(file)
            fidelity_loss_distribution = np.load(file)
        return cls(models,
                   labels_count,
                   scores_for_surrogate_model,
                   losses_for_surrogate_model,
                   losses_mean_for_cv_model,
                   losses_std_for_cv_model,
                   fidelity_loss_on_explanation,
                   fidelity_loss_on_generated_data,
                   fidelity_loss_on_kfold_mean,
                   fidelity_loss_on_kfold_std,
                   fidelity_loss_distribution)

    def save_results(self,
                     filename):
        with open(f"{filename}.npy", "wb") as file:
            np.save(file, self.scores_for_surrogate_model)
            np.save(file, self.losses_for_surrogate_model)
            np.save(file, self.losses_mean_for_cv_model)
            np.save(file, self.losses_std_for_cv_model)
            np.save(file, self.fidelity_loss_on_explanation)
            np.save(file, self.fidelity_loss_on_generated_data)
            np.save(file, self.fidelity_loss_on_kfold_mean)
            np.save(file, self.fidelity_loss_on_kfold_std)
            np.save(file, self.fidelity_loss_distribution)

    def _plot_for_each_label(self,
                             data,
                             data_desc):
        for model_idx, (classifier_name, model) in enumerate(self.models):
            fig, axs = plt.subplots(nrows=1, ncols=self.labels_count)
            fig.set_figwidth(5 * self.labels_count)
            fig.set_figheight(4)
            fig.suptitle(
                f"Histogram {data_desc} \n wyjaśniany model: {classifier_name}", fontsize=16)
            fig.tight_layout(pad=2.0)
            for idx in range(self.labels_count):
                data_to_plot = data[model_idx, :, idx]
                mean_value = float(np.mean(data_to_plot))
                axs[idx].hist(data_to_plot, bins=30)
                axs[idx].axvline(mean_value, color="red", linestyle="--")
                axs[idx].set_title(f"#{idx} klasa | średnia={round(mean_value, 4)}")
            plt.show()

    def plot_scores_for_surrogate_model(self):
        self._plot_for_each_label(
            data=self.scores_for_surrogate_model,
            data_desc="miary score na części treningowej")

    def plot_losses_for_surrogate_model(self):
        self._plot_for_each_label(
            data=self.losses_for_surrogate_model,
            data_desc="MSE na części treningowej")

    def plot_losses_mean_for_cv_model(self):
        self._plot_for_each_label(
            data=self.losses_mean_for_cv_model,
            data_desc="MSE (CV - wartość średnia)")

    def plot_losses_std_for_cv_model(self):
        self._plot_for_each_label(
            data=self.losses_std_for_cv_model,
            data_desc="MSE (CV - odchylenie standardowe)")

    def _plot_for_each_model(self,
                             data,
                             data_desc):
        fig, axs = plt.subplots(nrows=len(self.models) // 3, ncols=3)
        fig.set_figwidth(15)
        fig.set_figheight(7)
        fig.suptitle(f"Histogram {data_desc}", fontsize=16)
        fig.tight_layout(pad=2.0)
        for model_idx, (classifier_name, model) in enumerate(self.models):
            idx_row = floor(model_idx / 3)
            idx_col = model_idx % 3
            data_to_plot = data[model_idx, :]
            mean_value = float(np.mean(data_to_plot))
            axs[idx_row][idx_col].hist(data_to_plot, bins=30)
            axs[idx_row][idx_col].axvline(mean_value, color="red", linestyle="--")
            axs[idx_row][idx_col].set_title(
                f"model wyjaśniany: {classifier_name} | średnia={round(mean_value, 4)}")
        plt.show()

    def plot_fidelity_loss_on_explanation(self):
        self._plot_for_each_model(
            data=self.fidelity_loss_on_explanation,
            data_desc="błędu odwzorowania na wyjaśnianym przykładzie")

    def plot_fidelity_losses_on_generated_data(self):
        self._plot_for_each_model(
            data=self.fidelity_loss_on_generated_data,
            data_desc="błędu odwzorowania na zbiorze syntetycznym (część treningowa)")

    def plot_fidelity_loss_on_kfold_mean(self):
        self._plot_for_each_model(
            data=self.fidelity_loss_on_kfold_mean,
            data_desc="błędu odwzorowania (CV - wartości średnie) na zbiorze syntetycznym")

    def plot_fidelity_loss_on_kfold_std(self):
        self._plot_for_each_model(
            data=self.fidelity_loss_on_kfold_std,
            data_desc="błędu odwzorowania (CV - odchylenie standardowe) na zbiorze syntetycznym")

    def plot_fidelity_loss_distribution(self, domain_unit="Quantiles"):
        fig, axs = plt.subplots(nrows=len(self.models) // 3, ncols=3)
        fig.set_figwidth(15)
        fig.set_figheight(7)
        fig.suptitle(f"Uśredniony rozkład błędu utraty odwzorowania wg odległości od wyjaśnianego przykładu", fontsize=16)
        fig.tight_layout(pad=3.0, h_pad=4.0)
        for model_idx, (classifier_name, model) in enumerate(self.models):
            idx_row = floor(model_idx / 3)
            idx_col = model_idx % 3
            axs[idx_row][idx_col].plot(np.mean(self.fidelity_loss_distribution[model_idx, :], axis=0))
            axs[idx_row][idx_col].set_title(f"wyjaśniany model: {classifier_name}")
        axs[0][0].set_ylabel(f"Miara utraty odwzorowania próbek")
        axs[1][0].set_ylabel(f"Miara utraty odwzorowania próbek")
        axs[1][0].set_xlabel(f"Kolejne {domain_unit} odległości od wyjaśnienego przykładu")
        axs[1][1].set_xlabel(f"Kolejne {domain_unit} odległości od wyjaśnienego przykładu")
        axs[1][2].set_xlabel(f"Kolejne {domain_unit} odległości od wyjaśnienego przykładu")
        plt.show()


class CompareResults:

    def __init__(self,
                 models):
        self.model_names = [model[0] for model in models]

    def plot_by_explained_model_and_variant(self,
                                            data_default,
                                            data_multiregressor,
                                            data_multiclassifier,
                                            data_singleclassifier,
                                            ylabel,
                                            title,
                                            ymin=None):

        labels = self.model_names
        label_locations = np.arange(len(labels))

        bar_width = 0.15
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.bar(label_locations - 1.5 * bar_width, data_default, bar_width, label="Wariant oryginalny")
        ax.bar(label_locations - 0.5 * bar_width, data_multiregressor, bar_width, label="Komitet drzew regresji")
        ax.bar(label_locations + 0.5 * bar_width, data_multiclassifier, bar_width, label="Komitet drzew decyzyjnych")
        ax.bar(label_locations + 1.5 * bar_width, data_singleclassifier, bar_width, label="Drzewo decyzyjne")

        if ymin is not None:
            ax.set_ylim(ymin=ymin)
        ax.set_xlabel("Wyjaśniany model")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(label_locations)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y')

        return fig
