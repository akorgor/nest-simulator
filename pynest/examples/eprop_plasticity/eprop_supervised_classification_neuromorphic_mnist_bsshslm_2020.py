# -*- coding: utf-8 -*-
#
# eprop_supervised_classification_neuromorphic_mnist_bsshslm_2020.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

r"""
Tutorial on learning N-MNIST classification with e-prop
-------------------------------------------------------

Training a classification model using supervised e-prop plasticity to classify the Neuromorphic MNIST (N-MNIST) dataset.

Description
~~~~~~~~~~~

This script demonstrates supervised learning of a classification task with the eligibility propagation (e-prop)
plasticity mechanism by Bellec et al. [1]_.

The primary objective of this task is to classify the N-MNIST dataset [2]_, an adaptation of the traditional
MNIST dataset of handwritten digits specifically designed for neuromorphic computing. The N-MNIST dataset
captures changes in pixel intensity through a dynamic vision sensor, converting static images into sequences of
binary events, which we interpret as spike trains. This conversion closely emulates biological neural
processing, making it a fitting challenge for an e-prop-equipped spiking neural network (SNN).

.. image:: eprop_supervised_classification_neuromorphic_mnist_bsshslm_2020.png
   :width: 70 %
   :alt: Schematic of network architecture. Same as Figure 1 in the code.
   :align: center

Learning in the neural network model is achieved by optimizing the connection weights with e-prop plasticity.
This plasticity rule requires a specific network architecture depicted in Figure 1. The neural network model
consists of a recurrent network that receives input from spike generators and projects onto multiple readout
neurons - one for each class. Each input generator is assigned to a pixel of the input image; when an event is
detected in a pixel at time :math:`t`, the corresponding input generator (connected to an input neuron) emits a spike
at that time. Each readout neuron compares the network signal :math:`\pi_k` with the target signal :math:`\pi_k^*`,
which it receives from a rate generator representing the respective digit class. The network's training error is
assessed by employing a cross-entropy error loss.

Details on the event-based NEST implementation of e-prop can be found in [3]_.

References
~~~~~~~~~~

.. [1] Bellec G, Scherr F, Subramoney F, Hajek E, Salaj D, Legenstein R, Maass W (2020). A solution to the
       learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11:3625.
       https://doi.org/10.1038/s41467-020-17236-y

.. [2] Orchard, G., Jayawant, A., Cohen, G. K., & Thakor, N. (2015). Converting static image datasets to
       spiking neuromorphic datasets using saccades. Frontiers in neuroscience, 9, 159859.

.. [3] Korcsak-Gorzo A, Espinoza Valverde JA, Stapmanns J, Plesser HE, Dahmen D,
       Bolten M, van Albada SJ, Diesmann M (2025). Event-driven eligibility
       propagation in large sparse networks: efficiency shaped by biological
       realism. arXiv:2511.21674. https://doi.org/10.48550/arXiv.2511.21674

"""  # pylint: disable=line-too-long # noqa: E501

# %% ###########################################################################################################
# Import libraries
# ~~~~~~~~~~~~~~~~
# We begin by importing all libraries required for the simulation, analysis, and visualization.

import zipfile
from pathlib import Path

import nest
import numpy as np
import requests
from mpi4py import MPI
from plotting import Plotter
from toolbox import Tools

# %% ###########################################################################################################
# Setup
# ~~~~~

cfg = dict(
    E_L=-0.05,
    E_L_out=-0.1,
    V_reset=-0.3,
    V_th=0.75,
    average_gradient=False,
    batch_size=2,
    c_reg=200.0,
    delete_existing_recordings=False,
    do_early_stopping=False,
    do_plotting=True,
    eta=3e-4,
    exc_neuron_fraction=0.5,
    f_target=10.0,
    job_cpus_per_task=1,
    job_nodes=1,
    job_ntasks_per_node=1,
    learning_window=300,
    loss="cross_entropy",
    n_iter_train_chunk=5,
    n_iter_test=1,
    n_iter_train=5,
    n_iter_validate_every=10,
    record_dynamics=True,
    record_weights=True,
    recurrent_connectivity=0.1,
    relative_path_data_dir="data",
    relative_path_figures_dir="figures",
    relative_path_recordings_dir="recordings",
    reset_neurons=True,
    save_weights=True,
    scale_weight_inp_rec=0.03,
    scale_weight_out_rec=1.5,
    scale_weight_rec_out=0.006,
    scale_weight_rec_rec=0.02,
    seed=1,
    stop_crit=0.05,
    surrogate_gradient="piecewise_linear",
    surrogate_gradient_height=20.0,
    surrogate_gradient_width=0.05,
    t_ref=2.0,
    tau_m=10.0,
    tau_m_out=50.0,
    verify=True,
    weight_dale_enforced_inp=False,
    weight_dale_enforced_out=False,
    weight_dale_enforced_rec=False,
    weight_sign_fixed_inp=False,
    weight_sign_fixed_out=False,
    weight_sign_fixed_rec=False,
)

tools = Tools(cfg, __file__)
cfg = tools.cfg

local_num_threads = cfg["job_cpus_per_task"]
total_num_virtual_procs = cfg["job_nodes"] * cfg["job_ntasks_per_node"] * local_num_threads

# %% ###########################################################################################################
# Schematic of network architecture
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This figure, identical to the one in the description, shows the required network architecture in the center,
# the input and output of the classification task above, and lists of the required NEST device, neuron, and
# synapse models below. The connections that must be established are numbered 1 to 7.

tools.show_image()

# %% ###########################################################################################################
# Initialize random generator
# ...........................
# We seed the numpy random generator, which will generate random initial weights as well as random input and
# output.

np.random.seed(cfg["seed"])  # fix numpy random seed

# %% ###########################################################################################################
# Define timing of task
# .....................
# The task's temporal structure is then defined, once as time steps and once as durations in milliseconds.
# Using a batch size larger than one aids the network in generalization, facilitating the solution to this task.
# The number of iterations for good convergence requires distributed computing. Increasing the number of
# iterations enhances learning performance up to the point where overfitting occurs. If early stopping is enabled, the
# classification error is tested in regular intervals and the training stopped as soon as the error selected as
# stop criterion is reached. After training, the performance can be tested over a number of test iterations.

batch_size = cfg["batch_size"]  # batch size, 100 for convergence
n_iter_train = cfg["n_iter_train"]  # number of training iterations, 200 for convergence
n_iter_train_chunk = cfg["n_iter_train_chunk"]  # chunk training iterations to reduce memory usage
n_iter_test = cfg["n_iter_test"]  # number of iterations for final test
do_early_stopping = cfg["do_early_stopping"]  # if True, stop training as soon as stop criterion fulfilled
n_iter_validate_every = cfg["n_iter_validate_every"]  # number of training iterations before validation
n_iter_validate = 1  # number of validation iterations to average over
n_iter_early_stop = 8  # number of iterations to average over to evaluate early stopping condition
stop_crit = cfg["stop_crit"]  # error value corresponding to stop criterion for early stopping

steps = dict(
    sequence=300,  # time steps of one full sequence
    learning_window=cfg["learning_window"],  # time steps of window with non-zero learning signals
    offset_gen=1,  # offset since generator signals start from time step 1
    delay_inp_rec=1,  # connection delay between input and recurrent neurons
    delay_rec_out=1,  # connection delay between recurrent and output neurons
    delay_out_norm=1,  # connection delay between output neurons for normalization
    extension_sim=1,  # extra time step to close right-open simulation time interval in Simulate()
    final_update=3,  # extra time steps to update all synapses at the end of task
)

steps["delays"] = sum(v for k, v in steps.items() if k.startswith("delay"))  # time steps of delays

steps["total_offset"] = steps["offset_gen"] + steps["delays"]  # time steps of total offset

duration = dict(step=1.0)  # ms, temporal resolution of the simulation

duration.update(dict((key, value * duration["step"]) for key, value in steps.items()))  # ms, durations

# %% ###########################################################################################################
# Set up simulation
# .................
# As last step of the setup, we reset the NEST kernel to remove all existing NEST simulation settings and
# objects and set some NEST kernel parameters, some of which are e-prop-related.

params_setup = dict(
    data_path=str(tools.path_recordings_dir),  # path to save data to
    eprop_learning_window=duration["learning_window"],
    eprop_reset_neurons_on_update=cfg[
        "reset_neurons"
    ],  # if True, reset dynamic variables at start of each update interval
    eprop_update_interval=duration["sequence"],  # ms, time interval for updating the synaptic weights
    local_num_threads=local_num_threads,
    overwrite_files=False,  # if True, overwrite existing files
    print_time=False,  # if True, print time progress bar during simulation, set False if run as code cell
    resolution=duration["step"],
    rng_seed=cfg["seed"],  # seed for NEST random number generator
    total_num_virtual_procs=total_num_virtual_procs,  # number of virtual processes, set in case of distributed computing
)

####################

nest.verbosity = nest.VerbosityLevel.FATAL
nest.ResetKernel()
nest.set(**params_setup)

comm = MPI.COMM_WORLD

# %% ###########################################################################################################
# Create neurons
# ~~~~~~~~~~~~~~
# We proceed by creating a certain number of input, recurrent, and readout neurons and setting their parameters.
# Additionally, we already create an input spike generator and an output target rate generator, which we will
# configure later. Each input sample is mapped out to a 34x34 pixel grid and a polarity dimension. We allocate
# spike generators to each input image pixel to simulate spike events.

pixels_dict = dict(
    n_x=34,  # number of pixels in horizontal direction
    n_y=34,  # number of pixels in vertical direction
    n_polarity=2,  # number of pixels in the dimension coding for polarity
    time_max=336040,  # in microseconds, longest recording over training and test set
)

pixels_dict["n_total"] = pixels_dict["n_x"] * pixels_dict["n_y"] * pixels_dict["n_polarity"]  # total number of pixels

n_in = pixels_dict["n_total"]  # number of input neurons = 1196
n_rec = 150  # number of recurrent neurons
n_out = 10  # number of readout neurons

params_nrn_out = dict(
    C_m=1.0,  # pF, membrane capacitance - takes effect only if neurons get current input (here not the case)
    E_L=cfg["E_L_out"],  # mV, leak / resting membrane potential
    I_e=0.0,  # pA, external current input
    loss=cfg["loss"],  # loss function
    regular_spike_arrival=False,  # If True, input spikes arrive at end of time step, if False at beginning
    tau_m=cfg["tau_m_out"],  # ms, membrane time constant
    V_m=0.0,  # mV, initial value of the membrane voltage
)

params_nrn_rec = dict(
    C_m=1.0,
    c_reg=cfg["c_reg"],  # coefficient of firing rate regularization
    E_L=cfg["E_L"],
    f_target=cfg["f_target"],  # spikes/s, target firing rate for firing rate regularization
    I_e=0.0,
    regular_spike_arrival=True,
    surrogate_gradient_function=cfg["surrogate_gradient"],  # surrogate gradient / pseudo-derivative function
    surrogate_gradient_height=cfg["surrogate_gradient_height"],  # height scaling of the pseudo-derivative
    surrogate_gradient_width=cfg["surrogate_gradient_width"],  # width scaling of the pseudo-derivative
    t_ref=cfg["t_ref"],  # ms, duration of refractory period
    tau_m=cfg["tau_m"],
    V_m=0.0,
    V_th=cfg["V_th"],  # mV, spike threshold membrane voltage
    flush_event_send_interval=duration["sequence"],
)

####################

# Intermediate parrot neurons required between input spike generators and recurrent neurons,
# since devices cannot establish plastic synapses for technical reasons

gen_spk_in = nest.Create("spike_generator", n_in)
nrns_inp = nest.Create("parrot_neuron", n_in)

# The suffix _bsshslm_2020 follows the NEST convention to indicate in the model name the paper
# that introduced it by the first letter of the authors' last names and the publication year.

nrns_rec = nest.Create("eprop_iaf_bsshslm_2020", n_rec, params_nrn_rec)
nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
gen_rate_target = nest.Create("step_rate_generator", n_out)

# %% ###########################################################################################################
# Create recorders
# ~~~~~~~~~~~~~~~~
# We also create recorders, which, while not required for the training, will allow us to track various dynamic
# variables of the neurons, spikes, and changes in synaptic weights. To save computing time and memory, the
# recorders, the recorded variables, neurons, and synapses can be limited to the ones relevant to the
# experiment, and the recording interval can be increased (see the documentation on the specific recorders). By
# default, recordings are stored in memory but can also be written to file.

n_record = 1  # number of neurons to record dynamic variables from - this script requires n_record >= 1
n_record_w = 5  # number of senders and targets to record weights from - this script requires n_record_w >=1

if n_record == 0 or n_record_w == 0:
    raise ValueError("n_record and n_record_w >= 1 required")

params_mm_rec = dict(
    interval=duration["step"],  # interval between two recorded time points
    record_from=["V_m", "surrogate_gradient", "learning_signal"],  # dynamic variables to record
    start=duration["offset_gen"] + duration["delay_inp_rec"],  # start time of recording
    label="multimeter_rec",
    record_to="ascii",
    precision=16,
)

params_mm_out = dict(
    interval=duration["step"],
    record_from=["readout_signal", "target_signal"],
    start=duration["total_offset"],
    label="multimeter_out",
    record_to="ascii",
    precision=16,
)

params_wr = dict(
    start=duration["total_offset"],
    label="weight_recorder",
    record_to="ascii",
    precision=16,
)

params_sr_in = dict(
    start=duration["offset_gen"],
    label="spike_recorder_in",
    record_to="ascii",
    precision=16,
)

params_sr_rec = dict(
    start=duration["offset_gen"],
    label="spike_recorder_rec",
    record_to="ascii",
    precision=16,
)

####################

if cfg["record_dynamics"]:
    params_mm_out["record_from"] += ["V_m", "readout_signal_unnorm", "error_signal"]

    mm_rec = nest.Create("multimeter", params_mm_rec)
    sr_in = nest.Create("spike_recorder", params_sr_in)
    sr_rec = nest.Create("spike_recorder", params_sr_rec)

if cfg["record_weights"]:
    wr = nest.Create("weight_recorder", params_wr)

mm_out = nest.Create("multimeter", params_mm_out)

nrns_rec_record = nrns_rec[:n_record]

# %% ###########################################################################################################
# Force final update
# ~~~~~~~~~~~~~~~~~~
# Synapses only get active, that is, the correct weight update calculated and applied, when they transmit a
# spike. To still be able to read out the correct weights at the end of the simulation, we force spiking of the
# presynaptic neuron and thus an update of all synapses, including those that have not transmitted a spike in
# the last update interval, by sending a strong spike to all neurons that form the presynaptic side of an eprop
# synapse. This step is required purely for technical reasons.

gen_spk_final_update = nest.Create("spike_generator", 1)

# %% ###########################################################################################################
# Create connections
# ~~~~~~~~~~~~~~~~~~
# Now, we define the connectivity and set up the synaptic parameters, with the synaptic weights drawn from
# random distributions. After these preparations, we establish the enumerated connections of the core network,
# as well as additional connections to the recorders.

params_conn_all_to_all = dict(rule="all_to_all", allow_autapses=False)
params_conn_one_to_one = dict(rule="one_to_one")

params_common_syn_eprop = dict(
    optimizer=dict(
        type="gradient_descent",  # algorithm to optimize the weights
        batch_size=1,
        Wmin=-100.0,  # pA, minimal limit of the synaptic weights
        Wmax=100.0,  # pA, maximal limit of the synaptic weights
    ),
    average_gradient=cfg["average_gradient"],  # if True, average the gradient over the learning window
)

eta_test = 0.0  # learning rate for test phase
eta_train = cfg["eta"]  # learning rate for training phase

plastic_synapse_model = "eprop_synapse_bsshslm_2020"

params_syn_in = dict(
    synapse_model=plastic_synapse_model,
    delay=duration["step"],
    tau_m_readout=params_nrn_out["tau_m"],
    weight=nest.random.normal(std=cfg["scale_weight_inp_rec"]),  # pA, initial values for the synaptic weights
)

params_syn_rec = dict(
    synapse_model=plastic_synapse_model,
    delay=duration["step"],
    tau_m_readout=params_nrn_out["tau_m"],
    weight=nest.random.normal(std=cfg["scale_weight_rec_rec"]),
)

params_syn_out = dict(
    synapse_model=plastic_synapse_model,
    delay=duration["step"],
    tau_m_readout=params_nrn_out["tau_m"],
    weight=nest.random.uniform(min=-cfg["scale_weight_rec_out"], max=cfg["scale_weight_rec_out"]),
)

params_syn_feedback = dict(
    synapse_model="eprop_learning_signal_connection_bsshslm_2020",
    delay=duration["step"],
    weight=nest.random.normal(std=cfg["scale_weight_out_rec"]),
)

params_syn_out_out = dict(
    synapse_model="rate_connection_delayed",
    delay=duration["step"],
    receptor_type=1,  # receptor type of readout neuron to receive other readout neuron's signals for softmax
    weight=1.0,  # pA, weight 1.0 required for correct softmax computation for technical reasons
)

params_syn_rate_target = dict(
    synapse_model="rate_connection_delayed",
    delay=duration["step"],
    receptor_type=2,  # receptor type over which readout neuron receives target signal
)

params_syn_static = dict(
    synapse_model="static_synapse",
    delay=duration["step"],
)

####################

nest.SetDefaults(plastic_synapse_model, params_common_syn_eprop)

nest.Connect(gen_spk_in, nrns_inp, params_conn_one_to_one, params_syn_static)  # connection 1
nest.Connect(nrns_inp, nrns_rec, params_conn_all_to_all, params_syn_in)  # connection 2
nest.Connect(
    nrns_rec,
    nrns_rec,
    dict(
        rule="fixed_indegree",
        indegree=int(cfg["recurrent_connectivity"] * n_rec),
        allow_multapses=False,
        allow_autapses=False,
    ),
    params_syn_rec,
)  # connection 3
nest.Connect(nrns_rec, nrns_out, params_conn_all_to_all, params_syn_out)  # connection 4
nest.Connect(nrns_out, nrns_rec, params_conn_all_to_all, params_syn_feedback)  # connection 5
nest.Connect(gen_rate_target, nrns_out, params_conn_one_to_one, params_syn_rate_target)  # connection 6
nest.Connect(nrns_out, nrns_out, params_conn_all_to_all, params_syn_out_out)  # connection 7
nest.Connect(gen_spk_final_update, nrns_inp + nrns_rec, "all_to_all", dict(weight=1000.0))
nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

if cfg["record_dynamics"]:
    nest.Connect(nrns_inp, sr_in, params_conn_all_to_all, params_syn_static)
    nest.Connect(nrns_rec, sr_rec, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_rec, nrns_rec_record, params_conn_all_to_all, params_syn_static)

if cfg["record_weights"]:
    tools.configure_weight_recorder_connections(wr, nrns_inp, nrns_rec, nrns_out, n_record_w)
    nest.SetDefaults(plastic_synapse_model, dict(weight_recorder=wr))

tools.constrain_weights(nrns_inp, nrns_rec, params_syn_in, "inp")
tools.constrain_weights(nrns_rec, nrns_rec, params_syn_rec, "rec")
tools.constrain_weights(nrns_rec, nrns_out, params_syn_out, "out")

# %% ###########################################################################################################
# Create input and output
# ~~~~~~~~~~~~~~~~~~~~~~~
# This section involves downloading the N-MNIST dataset, extracting it, and preparing it for neural network
# training and testing. The dataset consists of two main components: training and test sets.

# The `download_and_extract_nmnist_dataset` function retrieves the dataset from its public repository and
# extracts it into a specified directory. It checks for the presence of the dataset to avoid re-downloading.
# After downloading, it extracts the main dataset zip file, followed by further extraction of nested zip files
# for training and test data, ensuring that the dataset is ready for loading and processing.

# The `load_image` function reads a single image file from the dataset, converting the event-based neuromorphic
# data into a format suitable for processing by spiking neural networks. It arranges the events into a
# structured format representing the image.

# The `DataLoader` class facilitates the loading of the dataset for neural network training and testing. It
# supports selecting specific labels for inclusion, allowing for targeted training on subsets of the dataset.
# The class also includes functionality for random shuffling and batching of data, ensuring that diverse and
# representative samples are used throughout the training process.


def unzip(zip_file_path, extraction_path):
    print(f"Extracting {zip_file_path}.")
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        zip_file.extractall(extraction_path)
    zip_file_path.unlink()


def download_and_extract_nmnist_dataset(save_path):
    nmnist_dataset = dict(
        url="https://data.mendeley.com/public-api/zip/468j46mzdv/download/1",
        directory="468j46mzdv-1",
        zip="dataset.zip",
    )

    save_path = Path(save_path)
    path = save_path / nmnist_dataset["directory"]

    train_path = path / "Train"
    test_path = path / "Test"

    downloaded_zip_path = save_path / nmnist_dataset["zip"]

    if not (path.exists() and train_path.exists() and test_path.exists()):
        if not downloaded_zip_path.exists():
            print("\nDownloading the N-MNIST dataset.")
            chunk_size = 1024 * 1024  # 1 MiB
            with requests.get(nmnist_dataset["url"], stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(downloaded_zip_path, "wb", buffering=chunk_size) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

        unzip(downloaded_zip_path, save_path)
        unzip(train_path.with_suffix(".zip"), path)
        unzip(test_path.with_suffix(".zip"), path)

    return train_path, test_path


def load_image(file_path, pixels_dict):
    with open(file_path, "rb") as f:
        byte_array = np.frombuffer(f.read(), dtype=np.uint8)

    n_byte_columns = 5
    byte_array = byte_array.reshape(-1, n_byte_columns)

    x_coords = byte_array[:, 0].astype(np.int64)  # in pixels
    y_coords = byte_array[:, 1].astype(np.int64)  # in pixels

    byte2 = byte_array[:, 2].astype(np.uint64)
    byte3 = byte_array[:, 3].astype(np.uint64)
    byte4 = byte_array[:, 4].astype(np.uint64)

    polarities = (byte2 >> 7).astype(np.int64)  # 0 for OFF, 1 for ON

    mask_22_bit = np.uint64(0x7FFFFF)  # mask to keep only lower 22 bits
    times = (((byte2 & 0x7F) << 16) | (byte3 << 8) | byte4) & mask_22_bit
    times = times.astype(np.int64)  # in microseconds
    times = np.around(times * duration["sequence"] / pixels_dict["time_max"])  # map sample to sequence length

    pixel_index = polarities * pixels_dict["n_x"] * pixels_dict["n_y"] + y_coords * pixels_dict["n_x"] + x_coords

    sort_idx = np.lexsort((times, pixel_index))  # sort events first by pixel index, then by time
    times_sorted = times[sort_idx]
    pixels_sorted = pixel_index[sort_idx]

    all_pixel_indices = np.arange(pixels_dict["n_total"])

    # find, for each pixel index, its insertion point in the sorted pixel array, producing the batch boundaries
    pixel_boundaries = np.searchsorted(pixels_sorted, all_pixel_indices)

    # split the sorted times at the pixel boundaries; skip the first boundary to avoid an empty initial segment
    image = np.split(times_sorted, pixel_boundaries[1:])
    return image


class DataLoader:
    def __init__(self, path, selected_labels, batch_size, pixels_dict):
        self.path = Path(path)
        self.selected_labels = selected_labels
        self.batch_size = batch_size
        self.pixels_dict = pixels_dict

        self.current_index = 0
        self.set_all_sample_paths_with_labels()
        self.n_all_samples = len(self.all_sample_paths)
        self.shuffled_indices = np.random.permutation(self.n_all_samples)

    def set_all_sample_paths_with_labels(self):
        self.all_sample_paths = []
        self.all_labels = []

        for label in self.selected_labels:
            for sample in sorted((self.path / str(label)).iterdir()):
                self.all_sample_paths.append(sample.absolute())
                self.all_labels.append(label)

    def get_new_evaluation_batch(self):
        end_index = self.current_index + self.batch_size

        selected_indices = np.take(self.shuffled_indices, range(self.current_index, end_index), mode="wrap")

        self.current_index = (self.current_index + self.batch_size) % self.n_all_samples

        images_batch = [load_image(self.all_sample_paths[i], self.pixels_dict) for i in selected_indices]
        labels_batch = [self.all_labels[i] for i in selected_indices]

        return images_batch, labels_batch


def get_params_task_input_output(n_iter_interval, n_iter_curr, loader):
    iteration_offset = n_iter_interval * batch_size * duration["sequence"]

    spike_times = [[] for _ in range(n_in)]

    params_gen_rate_target = [
        dict(
            amplitude_times=np.arange(0.0, n_iter_curr * batch_size * duration["sequence"], duration["sequence"])
            + iteration_offset
            + duration["total_offset"],
            amplitude_values=np.zeros(n_iter_curr * batch_size),
        )
        for _ in range(n_out)
    ]

    for i in range(n_iter_curr):
        input_batch, target_batch = loader.get_new_evaluation_batch()
        for batch_element in range(batch_size):
            params_gen_rate_target[target_batch[batch_element]]["amplitude_values"][
                i * batch_size + batch_element
            ] = 1.0

            for n, relative_times in enumerate(input_batch[batch_element]):
                if len(relative_times) > 0:
                    relative_times = np.array(relative_times)
                    spike_times[n].extend(
                        iteration_offset
                        + (i * batch_size + batch_element) * duration["sequence"]
                        + relative_times
                        + duration["offset_gen"]
                    )

    params_gen_spk_in = [dict(spike_times=spk_times) for spk_times in spike_times]

    return params_gen_spk_in, params_gen_rate_target


train_path, test_path = download_and_extract_nmnist_dataset(tools.path_data_dir)

selected_labels = [label for label in range(n_out)]

data_loader_train = DataLoader(train_path, selected_labels, batch_size, pixels_dict)
data_loader_test = DataLoader(test_path, selected_labels, batch_size, pixels_dict)

# %% ###########################################################################################################
# Save pre-training weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Before we begin training, we read out the initial weight matrices so that we can eventually compare them to
# the optimized weights.

if comm.rank == 0 and cfg["save_weights"]:
    tools.save_weights(nrns_inp, nrns_rec, "pre_train_inp")
    tools.save_weights(nrns_rec, nrns_rec, "pre_train_rec")
    tools.save_weights(nrns_rec, nrns_out, "pre_train_out")

# %% ###########################################################################################################
# Simulate and evaluate
# ~~~~~~~~~~~~~~~~~~~~~
# We train the network by simulating for a number of training iterations with the set learning rate. If early
# stopping is turned on, we evaluate the network's performance on the validation set in regular intervals and,
# if the error is below a certain threshold, we stop the training early. If the error is not below the
# threshold, we continue training until the end of the set number of iterations. Finally, we evaluate the
# network's performance on the test set.
# Furthermore, we evaluate the network's training error by calculating a loss - in this case, the cross-entropy
# error between the integrated recurrent network activity and the target rate.


class TrainingPipeline:
    def __init__(self):
        self.n_iter_sim = 0
        self.phase_label_previous = ""
        self.prefix_previous = ""
        self.error = 1.0
        self.k_iter = 0
        self.early_stop = False
        self.evaluate_curr = False

    def evaluate(self, events):
        senders, readout_signal, target_signal = events
        order = np.argsort(senders, kind="stable")

        seq = steps["sequence"]
        lw = steps["learning_window"]

        readout_signal = readout_signal[order].reshape((n_out, -1, batch_size, seq))[:, :, :, -lw:]
        target_signal = target_signal[order].reshape((n_out, -1, batch_size, seq))[:, :, :, -lw:]

        if cfg["loss"] == "cross_entropy":
            eps = np.float32(1e-7)
            r = np.clip(readout_signal, eps, 1.0)
            loss = -np.mean(np.sum(target_signal * np.log(r), axis=0), axis=(1, 2))
        elif cfg["loss"] == "mean_squared_error":
            diff = readout_signal - target_signal
            loss = 0.5 * np.mean(np.sum(diff * diff, axis=3), axis=(0, 2))

        r_mean = np.mean(readout_signal, axis=3)
        t_mean = np.mean(target_signal, axis=3)
        y_pred = np.argmax(r_mean, axis=0)
        y_true = np.argmax(t_mean, axis=0)

        errors = 1.0 - np.mean((y_true == y_pred), axis=1)
        error = float(np.mean(errors))

        tools.save_performance(self.n_iter_sim - self.n_iter, loss, errors, self.phase_label_previous)
        return error

    def run_phase(self, phase_label, eta, n_iter, evaluate=False):
        if n_iter == 0:
            return
        tools.set_synapse_defaults(eta)

        if phase_label == "training":
            loader = data_loader_train
        else:
            loader = data_loader_test

        params_gen_spk_in, params_gen_rate_target = get_params_task_input_output(self.n_iter_sim, n_iter, loader)
        gen_spk_in.set(params_gen_spk_in)
        gen_rate_target.set(params_gen_rate_target)

        self.process()
        self.evaluate_curr = evaluate

        self.prefix_previous = f"{(self.n_iter_sim+1):05d}_{phase_label}"
        self.simulate(
            n_iter * batch_size * duration["sequence"] - duration["total_offset"] - duration["extension_sim"],
            f"{self.prefix_previous}_0_",
        )

        self.n_iter = n_iter
        self.n_iter_sim += n_iter
        self.phase_label_previous = phase_label

    def simulate(self, duration, data_prefix=""):
        nest.data_prefix = data_prefix
        nest.Simulate(duration)

    def run(self):
        if do_early_stopping:
            for self.k_iter in range(0, n_iter_train, n_iter_validate_every):
                self.run_phase("validation", eta_test, n_iter_validate, True)
                self.run_phase("burn", eta_test, 1, True)
                if self.k_iter > 0 and self.error < stop_crit:
                    self.run_phase("early-stopping", eta_test, n_iter_early_stop, True)
                    self.run_phase("burn", eta_test, 1, True)
                    if self.error < stop_crit:
                        break
                self.run_phase("training", eta_train, n_iter_validate_every, True)
        else:
            for self.k_iter in range(0, n_iter_train, n_iter_train_chunk):
                self.run_phase("training", eta_train, n_iter_train_chunk, True)

        self.run_phase("test", eta_test, n_iter_test, True)

        self.process()

    def process(self):
        data_prefix = f"{self.prefix_previous}_1_" if self.n_iter_sim > 0 else f"{self.n_iter_sim:05d}_offset_0_"
        self.simulate(duration["total_offset"] + duration["extension_sim"], data_prefix)

        error = None

        if comm.rank == 0:
            if self.evaluate_curr:
                error = self.evaluate(tools.get_events(self.prefix_previous, save=True))
            else:
                tools.clear_events(self.prefix_previous)

        if self.evaluate_curr:
            self.error = comm.bcast(error, root=0)

    def evaluate_final(self):
        duration["task"] = self.n_iter_sim * batch_size * duration["sequence"]
        duration["sim"] = duration["task"] + duration["total_offset"] + duration["extension_sim"]

        gen_spk_final_update.set(dict(spike_times=[duration["sim"] + 1.0]))

        self.simulate(duration["final_update"])
        duration["sim"] += duration["final_update"]


training_pipeline = TrainingPipeline()
training_pipeline.run()
training_pipeline.evaluate_final()

if comm.rank != 0:
    exit()

if cfg["verify"]:
    tools.verify()

tools.save_kernel_status(nest.GetKernelStatus())
tools.save_node_ids(
    {
        "gen_spk_in": gen_spk_in,
        "nrns_inp": nrns_inp,
        "nrns_rec": nrns_rec,
        "nrns_out": nrns_out,
        "gen_rate_target": gen_rate_target,
        "gen_spk_final_update": gen_spk_final_update,
    }
)
tools.save_recordings("multimeter_out", duration)

# %% ###########################################################################################################
# Save recordings
# ~~~~~~~~~~~~~~~
# We can also retrieve the recorded history of the dynamic variables and weights, as well as detected spikes.

if cfg["record_dynamics"]:
    tools.save_recordings("multimeter_rec", duration)
    tools.save_recordings("spike_recorder_in", duration)
    tools.save_recordings("spike_recorder_rec", duration)

if cfg["record_weights"]:
    tools.save_recordings("weight_recorder", duration)

# %% ###########################################################################################################
# Save post-training weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# After the training, we can read out the optimized final weights.

if cfg["save_weights"]:
    tools.save_weights(nrns_inp, nrns_rec, "post_train_inp")
    tools.save_weights(nrns_rec, nrns_rec, "post_train_rec")
    tools.save_weights(nrns_rec, nrns_out, "post_train_out")

# %% ###########################################################################################################
# Plot results
# ~~~~~~~~~~~~
# Then, we plot a series of plots.

if cfg["do_plotting"]:
    data = tools.load_data()
    Plotter(
        __file__,
        cfg["relative_path_figures_dir"],
        data,
        duration["task"],
        duration["sequence"],
        steps["sequence"],
        batch_size,
        n_rec,
        n_out,
        cfg["record_dynamics"],
        include_plot_pattern=False,
    ).plot_all()
