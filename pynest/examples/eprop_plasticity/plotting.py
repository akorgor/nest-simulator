from collections import OrderedDict as odict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


class Plotter:
    def __init__(
        self,
        file_path,
        relative_path_figures_dir,
        data,
        duration_task,
        duration_sequence,
        steps_sequence,
        batch_size,
        n_rec,
        n_out,
        record_dynamics,
        include_plot_pattern,
    ):
        self.path_figures_dir = (Path(file_path).parent / relative_path_figures_dir).resolve()
        self.path_figures_dir.mkdir(parents=True, exist_ok=True)
        self.data = data
        self.duration_task = duration_task
        self.duration_sequence = duration_sequence
        self.steps_sequence = steps_sequence
        self.batch_size = batch_size
        self.n_rec = n_rec
        self.n_out = n_out
        self.record_dynamics = record_dynamics
        self.include_plot_pattern = include_plot_pattern

        self.colors = odict(
            blue="#2854c5",
            red="#e04b40",
            green="#25aa2c",
            gold="#f9c643",
            gray="#696969",
            orange="#f8933d",
            black="#000000",
            purple="#6f1970",
            lightorange="#fab377",
            lightred="#e98179",
            lightgreen="#66c36b",
            lightgray="#d3d3d3",
            lightblue="#b8dcfd",
            lightyellow="#fcdd91",
            mediumgray="#b8b8b8",
            pink="#fe5895",
            yellow="#f9f871",
        )

        plt.rcParams.update(
            dict(
                [
                    ("axes.spines.right", False),
                    ("axes.spines.top", False),
                    ("axes.prop_cycle", cycler(color=list(self.colors.values()))),
                    ("figure.dpi", 300),
                    ("font.family", "DejaVu Sans"),
                ]
            )
        )

    def plot_pattern(self):
        """
        Visualize the generated pattern alongside the target pattern for comparison. The two readout neurons encode the horizontal and vertical coordinates of the pattern, respectively.
        """

        idc = (self.data["multimeter_out_subset"].time >= self.duration_task - self.duration_sequence) & (
            self.data["multimeter_out_subset"].time < self.duration_task
        )

        order = np.argsort(self.data["multimeter_out_subset"].sender[idc].to_numpy(), kind="stable")

        readout_signal = (
            self.data["multimeter_out_subset"]
            .readout_signal[idc]
            .to_numpy()[order]
            .reshape(self.n_out, 1, self.batch_size, self.steps_sequence)
        )
        target_signal = (
            self.data["multimeter_out_subset"]
            .target_signal[idc]
            .to_numpy()[order]
            .reshape(self.n_out, 1, self.batch_size, self.steps_sequence)
        )

        fig, ax = plt.subplots()

        ax.plot(target_signal[0, 0, 0], -target_signal[1, 0, 0], c=self.colors["blue"], label="Target signal")
        ax.plot(
            readout_signal[0, 0, 0], -readout_signal[1, 0, 0], c=self.colors["red"], ls="--", label="Readout signal"
        )

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlabel("Signal 0")
        ax.set_ylabel("Signal 1")
        ax.axis("equal")

        fig.tight_layout()
        fig.savefig(self.path_figures_dir / "fig_pattern.pdf")

    def plot_learning_performance(self):
        """
        Plot the learning performance over time.
        """

        phases = ["validation", "training", "early-stopping", "test"]

        perf_list = [k for k in ["loss", "error"] if k in self.data["learning_performance"].columns]
        n_rows = len(perf_list)

        fig, axs = plt.subplots(n_rows, 1, sharex=True, figsize=(6, n_rows * 2))

        if n_rows == 1:
            axs = [axs]

        for ax, col in zip(axs, perf_list):
            for phase in phases:
                label = (phase[0].upper() + phase[1:]).replace("-", " ")
                idc = self.data["learning_performance"].phase == phase
                ax.scatter(
                    self.data["learning_performance"].iteration[idc],
                    self.data["learning_performance"][col][idc],
                    label=label,
                    marker="x",
                )

            ax.set_ylabel(col[0].upper() + col[1:])

        axs[-1].set_xlabel("Iteration")
        axs[-1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        axs[-1].xaxis.get_major_locator().set_params(integer=True)

        fig.align_ylabels()
        fig.tight_layout()
        fig.savefig(self.path_figures_dir / "fig_learning_performance.pdf")

    def plot_weight_matrices(self):
        """
        Plot the synaptic weight matrices before and after training.
        """

        fig, axs = plt.subplots(3, 2, figsize=(8, 7), sharex="col", sharey="row")

        df_w_list = [
            self.data["weights_pre_train_inp"],
            self.data["weights_post_train_inp"],
            self.data["weights_pre_train_rec"],
            self.data["weights_post_train_rec"],
            self.data["weights_pre_train_out"],
            self.data["weights_post_train_out"],
        ]

        w_abs_max = np.max([df.weight.abs().max() for df in df_w_list])
        args = dict(
            cmap=mpl.colors.LinearSegmentedColormap.from_list(
                "cmap", ((0.0, self.colors["blue"]), (0.5, "white"), (1.0, self.colors["red"]))
            ),
            vmin=-w_abs_max,
            vmax=w_abs_max,
        )

        for ax, df in zip(axs.flat, df_w_list):
            weight_matrix = df.pivot(index="sender", columns="receiver", values="weight").values
            if weight_matrix.shape[1] != self.n_rec:
                weight_matrix = weight_matrix.T
            cmesh = ax.pcolormesh(weight_matrix, **args)

        axs[2, 0].set_xlabel("Recurrent\nneurons")
        axs[2, 1].set_xlabel("Recurrent\nneurons")
        axs[0, 0].set_ylabel("Input\nneurons")
        axs[1, 0].set_ylabel("Recurrent\nneurons")
        axs[2, 0].set_ylabel("Readout\nneurons")
        fig.align_ylabels(axs[:, 0])

        axs[0, 0].text(0.5, 1.1, "Before training", transform=axs[0, 0].transAxes, ha="center")
        axs[0, 1].text(0.5, 1.1, "After training", transform=axs[0, 1].transAxes, ha="center")

        axs[2, 0].yaxis.get_major_locator().set_params(integer=True)

        plt.colorbar(cmesh, cax=axs[1, 1].inset_axes([1.1, 0.2, 0.05, 0.8]), label="Weight (pA)")

        fig.tight_layout()
        fig.savefig(self.path_figures_dir / "fig_weight-matrices.pdf")

    def build_recordables_list(self):
        """
        Construct the list of recorded variables and spike trains to be plotted.
        """

        candidates = [
            "spike_recorder_in_subset",
            "spike_recorder_rec_subset",
            "multimeter_rec_subset",
            "spike_recorder_reg_subset",
            "multimeter_reg_subset",
            "spike_recorder_ad_subset",
            "multimeter_ad_subset",
            "multimeter_out_subset",
        ]

        column_ylabels_list = [
            ("V_m", f"Membrane\nvoltage\n(mV)"),
            ("surrogate_gradient", f"Surrogate\ngradient\n"),
            ("V_th_adapt", f"Adaptive\nthreshold\n(mV)"),
            ("learning_signal", f"Learning\nsignal\n(pA)"),
            ("readout_signal", f"Readout\nsignal\n"),
            ("target_signal", f"Target\nsignal\n"),
            ("error_signal", f"Error\nsignal\n"),
        ]
        recordables_list = []
        for name in candidates:
            if name in self.data.keys():
                if name.startswith("spike_recorder"):
                    recordables_list.append((name, "spikes", f"Spikes\n\n"))
                elif name.startswith("multimeter"):
                    for column, ylabel in column_ylabels_list:
                        if column in self.data[name].columns:
                            recordables_list.append((name, column, ylabel))
        return recordables_list

    def plot_recordable(self, ax, xlims, name, recordable, ylabel):
        """
        Plot a selected recorded variable within the specified time window.
        """
        df = self.data[name]
        idc_times = (df.time >= xlims[0]) & (df.time < xlims[1])

        if recordable == "spikes":
            senders_subset = df.sender[idc_times]
            times_subset = df.time[idc_times]

            ax.scatter(times_subset, senders_subset, s=0.1, marker=".")

            right_ylabel = f"Neurons\n{senders_subset.min()}-{senders_subset.max()}"

            if len(senders_subset) > 0:
                y_min = np.min(senders_subset)
                y_max = np.max(senders_subset)
                margin = np.abs(y_max - y_min) * 0.1
                if margin == 0:
                    margin = 1
                ax.set_ylim(y_min - margin, y_max + margin)
        else:
            values = df[recordable]
            for sender in np.unique(df.sender):
                idc_sender = df.sender == sender
                idc = idc_sender & idc_times
                ax.plot(df.time[idc], values[idc])

            right_ylabel = f"Neuron\n{sender}"

            y_min = np.min(values)
            y_max = np.max(values)
            margin = np.abs(y_max - y_min) * 0.1
            if margin == 0:
                margin = 1
            ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_ylabel(ylabel)
        ax.text(
            1.02,
            0.5,
            right_ylabel,
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="left",
            multialignment="center",
            color=self.colors["gray"],
        )

    def plot_recordables(self):
        """
        Plot the time courses of the selected dynamic variables and spikes in two time windows: one at the beginning of training and one at the end.
        """

        recordables_list = self.build_recordables_list()

        subset = dict(
            Before=(1, self.duration_sequence), After=(self.duration_task - self.duration_sequence, self.duration_task)
        )
        n_subplots = len(recordables_list)
        for title, xlims in subset.items():
            fig, axs = plt.subplots(
                n_subplots,
                1,
                sharex=True,
                figsize=(8, n_subplots),
                gridspec_kw=dict(hspace=0.5, left=0.2, right=0.90, bottom=0.05, top=0.95),
            )
            fig.suptitle(f"{title} training")

            for ax, recordable in zip(axs, recordables_list):
                self.plot_recordable(ax, xlims, *recordable)

            axs[-1].set_xlabel("Time (ms)")
            axs[-1].set_xlim(*xlims)

            fig.align_ylabels()
            fig.savefig(self.path_figures_dir / f"fig_recordables_{title.lower()}-training.pdf")

    def plot_weight_time_courses(self):
        """
        Plot the synaptic weight trajectories over time. Unlike the multimeter and spike recorder, the weight recorder only stores a weight value when the recorded synapse is activated by a spike, so the initial weight is added manually.
        """

        id_to_label = self.data["node_ids"].set_index("id")["label"].str.split("_").str[1]
        self.data["weight_recorder_subset"]["label"] = (
            self.data["weight_recorder_subset"]["sender"].map(id_to_label)
            + "_"
            + self.data["weight_recorder_subset"]["receiver"].map(id_to_label)
        )

        weight_matrix_labels = self.data["weight_recorder_subset"].label.unique()

        fig, axs = plt.subplots(len(weight_matrix_labels), 1, sharex=True, sharey=True, figsize=(5, 5))

        w_list = []
        for i, (df_w_pre_train, label, name) in enumerate(
            [
                [self.data["weights_pre_train_inp"], "inp_rec", "Input"],
                [self.data["weights_pre_train_rec"], "rec_rec", "Recurrent"],
                [self.data["weights_pre_train_out"], "rec_out", "Output"],
            ]
        ):

            group = self.data["weight_recorder_subset"][self.data["weight_recorder_subset"].label == label]
            for sender in np.unique(group.sender):
                for receiver in np.unique(group.receiver):
                    df_sub = group[(group.sender == sender) & (group.receiver == receiver)].sort_values(by=["time"])
                    initial_weight = df_w_pre_train[
                        (df_w_pre_train.sender == sender) & (df_w_pre_train.receiver == receiver)
                    ].weight.values
                    if len(initial_weight) == 1:
                        weights = [initial_weight[0]] + df_sub.weight.tolist()
                        times = [0.0] + df_sub.time.tolist()
                        axs[i].step(times, weights, c=self.colors["blue"])
                        w_list.append(np.max(np.abs(weights)))
            axs[i].set_ylabel(f"{name} weights\n(pA)")

        w_abs_max = np.max(w_list)
        axs[-1].set_ylim(-w_abs_max * 1.1, w_abs_max * 1.1)
        axs[-1].set_xlabel("Time (ms)")
        axs[-1].set_xlim(0, self.duration_task)

        fig.align_ylabels()
        fig.tight_layout()
        fig.savefig(self.path_figures_dir / "fig_weight-time-courses.pdf")

    def plot_all(self):
        """
        Generate all enabled plots from the loaded data.
        """
        if self.include_plot_pattern:
            self.plot_pattern()

        self.plot_learning_performance()

        if self.record_dynamics:
            self.plot_weight_matrices()
            self.plot_recordables()
            self.plot_weight_time_courses()
