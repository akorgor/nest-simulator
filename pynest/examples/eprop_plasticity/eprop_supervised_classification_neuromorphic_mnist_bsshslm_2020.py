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
at that time. Each readout neuron compares the network signal :math:`\pi_k` with the teacher signal :math:`\pi_k^*`,
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import numpy as np
import requests
from IPython.display import Image
from mpi4py import MPI
from toolbox import Tools

# %% ###########################################################################################################
# Schematic of network architecture
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This figure, identical to the one in the description, shows the required network architecture in the center,
# the input and output of the classification task above, and lists of the required NEST device, neuron, and
# synapse models below. The connections that must be established are numbered 1 to 7.

try:
    Image(filename="./eprop_supervised_classification_neuromorphic_mnist_bsshslm_2020.png")
except Exception:
    pass

# %% ###########################################################################################################
# Setup
# ~~~~~

cfg = dict(
    E_L=-0.1,
    E_L_out=-0.1,
    V_reset=-0.3,
    V_th=0.85,
    average_gradient=False,
    batch_size=100,
    c_reg=50.0,
    constrain_weights_dale_in=False,
    constrain_weights_dale_out=False,
    constrain_weights_dale_rec=False,
    constrain_weights_sign_in=False,
    constrain_weights_sign_out=False,
    constrain_weights_sign_rec=False,
    cpus_per_task=1,
    dataset_dir="./",
    do_early_stopping=True,
    do_plotting=True,
    eta=5e-3,
    exc_to_inh_ratio=1.0,
    f_target=10.0,
    learning_window=300,
    loss="cross_entropy",
    model_nrn_rec="eprop_iaf_bsshslm_2020",
    n_iter_test=10,
    n_iter_train=400,
    n_iter_validate_every=10,
    nodes=1,
    ntasks_per_node=1,
    record_dynamics=False,
    results_dir="./results",
    recurrent_connectivity=0.1,
    remove_results_dir=False,
    reset_neurons=True,
    save_weights=True,
    scale_weight_inp_rec=0.03,
    scale_weight_out_rec=10.0,
    scale_weight_rec_out=0.006,
    scale_weight_rec_rec=0.02,
    seed=1,
    stop_crit=0.07,
    surrogate_gradient="piecewise_linear",
    surrogate_gradient_beta=1.7,
    surrogate_gradient_gamma=0.5,
    t_ref=4.0,
    tau_m=10.0,
    tau_m_out=100.0,
)

tools = Tools(cfg, __file__)
cfg = tools.cfg

local_num_threads = cfg["cpus_per_task"]
total_num_virtual_procs = cfg["nodes"] * cfg["ntasks_per_node"] * local_num_threads

# %% ###########################################################################################################
# Initialize random generator
# ...........................
# We seed the numpy random generator, which will generate random initial weights as well as random input and
# output.

rng_seed = cfg["seed"]  # numpy random seed
np.random.seed(rng_seed)  # fix numpy random seed

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

steps["delays"] = steps["delay_inp_rec"] + steps["delay_rec_out"] + steps["delay_out_norm"]  # time steps of delays

steps["total_offset"] = steps["offset_gen"] + steps["delays"]  # time steps of total offset

duration = dict(step=1.0)  # ms, temporal resolution of the simulation

duration.update(dict((key, value * duration["step"]) for key, value in steps.items()))  # ms, durations

# %% ###########################################################################################################
# Set up simulation
# .................
# As last step of the setup, we reset the NEST kernel to remove all existing NEST simulation settings and
# objects and set some NEST kernel parameters, some of which are e-prop-related.

params_setup = dict(
    eprop_learning_window=duration["learning_window"],
    eprop_reset_neurons_on_update=cfg[
        "reset_neurons"
    ],  # if True, reset dynamic variables at start of each update interval
    eprop_update_interval=duration["sequence"],  # ms, time interval for updating the synaptic weights
    print_time=False,  # if True, print time progress bar during simulation, set False if run as code cell
    resolution=duration["step"],
    total_num_virtual_procs=total_num_virtual_procs,  # number of virtual processes, set in case of distributed computing
    local_num_threads=local_num_threads,
    overwrite_files=False,  # if True, overwrite existing files
    data_path=str(tools.recordings_dir),  # path to save data to
    rng_seed=rng_seed,  # seed for random number generator
)

####################

nest.set_verbosity("M_FATAL")
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
    beta=cfg["surrogate_gradient_beta"],  # width scaling of the pseudo-derivative
    C_m=1.0,
    c_reg=cfg["c_reg"],  # coefficient of firing rate regularization
    E_L=cfg["E_L"],
    f_target=cfg["f_target"],  # spikes/s, target firing rate for firing rate regularization
    gamma=cfg["surrogate_gradient_gamma"],  # height scaling of the pseudo-derivative
    I_e=0.0,
    regular_spike_arrival=True,
    surrogate_gradient_function=cfg["surrogate_gradient"],  # surrogate gradient / pseudo-derivative function
    t_ref=cfg["t_ref"],  # ms, duration of refractory period
    tau_m=cfg["tau_m"],
    V_m=0.0,
    V_th=cfg["V_th"],  # mV, spike threshold membrane voltage
)

####################

# Intermediate parrot neurons required between input spike generators and recurrent neurons,
# since devices cannot establish plastic synapses for technical reasons

gen_spk_in = nest.Create("spike_generator", n_in)
if "eprop_input_neuron" in nest.get("node_models"):
    nrns_inp = nest.Create("eprop_input_neuron", n_in)
else:
    nrns_inp = nest.Create("parrot_neuron", n_in)

# The suffix _bsshslm_2020 follows the NEST convention to indicate in the model name the paper
# that introduced it by the first letter of the authors' last names and the publication year.

nrns_rec = nest.Create(cfg["model_nrn_rec"], n_rec, params_nrn_rec)
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
    wr = nest.Create("weight_recorder", params_wr)

mm_out = nest.Create("multimeter", params_mm_out)

nrns_rec_record = nrns_rec[:n_record]

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
        batch_size=batch_size,
        Wmin=-100.0,  # pA, minimal limit of the synaptic weights
        Wmax=100.0,  # pA, maximal limit of the synaptic weights
    ),
    average_gradient=cfg["average_gradient"],  # if True, average the gradient over the learning window
)

eta_test = 0.0  # learning rate for test phase
eta_train = cfg["eta"]  # learning rate for training phase

params_syn_base = dict(
    synapse_model="eprop_synapse_bsshslm_2020",
    delay=duration["step"],  # ms, dendritic delay
    tau_m_readout=params_nrn_out["tau_m"],  # ms, for technical reasons pass readout neuron membrane time constant
)

params_syn_in = dict(
    synapse_model="eprop_synapse_bsshslm_2020",
    delay=duration["step"],
    weight=nest.random.normal(std=cfg["scale_weight_inp_rec"]),
)

params_syn_rec = dict(
    synapse_model="eprop_synapse_bsshslm_2020",
    delay=duration["step"],
    weight=nest.random.normal(std=cfg["scale_weight_rec_rec"]),
)

params_syn_out = dict(
    synapse_model="eprop_synapse_bsshslm_2020",
    delay=duration["step"],
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

if cfg["record_dynamics"]:
    nest.Connect(nrns_inp, sr_in, params_conn_all_to_all, params_syn_static)
    nest.Connect(nrns_rec, sr_rec, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_rec, nrns_rec_record, params_conn_all_to_all, params_syn_static)
nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

if cfg["record_dynamics"]:
    sender_list = []
    target_list = []
    for pop_pre, pop_post in [(nrns_inp, nrns_rec), (nrns_rec, nrns_rec), (nrns_rec, nrns_out)]:
        conn_dict = nest.GetConnections(pop_pre, pop_post).get(["source", "target"])
        i = 0
        while i < n_record_w:
            idx = np.random.randint(0, len(conn_dict["source"]))
            sender = conn_dict["source"][idx]
            target = conn_dict["target"][idx]
            if sender not in sender_list and target not in target_list:
                sender_list.append(sender)
                target_list.append(target)
                i += 1
    params_wr.update(dict(senders=np.sort(sender_list), targets=np.sort(target_list)))
    nest.SetStatus(wr, params_wr)
    params_common_syn_eprop["weight_recorder"] = wr

nest.SetDefaults("eprop_synapse_bsshslm_2020", params_common_syn_eprop)

tools.constrain_weights([nrns_inp, nrns_rec, nrns_out], params_syn_base, params_common_syn_eprop)

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


def download_and_extract_nmnist_dataset(save_path="./"):
    nmnist_dataset = dict(
        url="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/468j46mzdv-1.zip",
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
            response = requests.get(nmnist_dataset["url"], timeout=10)
            with open(downloaded_zip_path, "wb") as file:
                file.write(response.content)

        unzip(downloaded_zip_path, save_path)
        unzip(f"{train_path}.zip", path)
        unzip(f"{test_path}.zip", path)

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


save_path = cfg["dataset_dir"]  # path to save the N-MNIST dataset to
train_path, test_path = download_and_extract_nmnist_dataset(save_path)

selected_labels = [label for label in range(n_out)]

data_loader_train = DataLoader(train_path, selected_labels, batch_size, pixels_dict)
data_loader_test = DataLoader(test_path, selected_labels, batch_size, pixels_dict)

# %% ###########################################################################################################
# Force final update
# ~~~~~~~~~~~~~~~~~~
# Synapses only get active, that is, the correct weight update calculated and applied, when they transmit a
# spike. To still be able to read out the correct weights at the end of the simulation, we force spiking of the
# presynaptic neuron and thus an update of all synapses, including those that have not transmitted a spike in
# the last update interval, by sending a strong spike to all neurons that form the presynaptic side of an eprop
# synapse. This step is required purely for technical reasons.

gen_spk_final_update = nest.Create("spike_generator", 1)

nest.Connect(gen_spk_final_update, nrns_inp + nrns_rec, "all_to_all", dict(weight=1000.0))

# %% ###########################################################################################################
# Read out pre-training weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Before we begin training, we read out the initial weight matrices so that we can eventually compare them to
# the optimized weights.

if comm.rank == 0:
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

        _, inv = np.unique(senders, return_inverse=True)
        n_senders = int(inv.max()) + 1 if inv.size else 0
        if n_senders == 0:
            return float(self.error)

        order = np.argsort(inv, kind="stable")
        readout_signal = readout_signal[order].reshape(n_senders, -1)
        target_signal = target_signal[order].reshape(n_senders, -1)

        seq = steps["sequence"]
        lw = steps["learning_window"]

        readout_signal = readout_signal.reshape((n_out, -1, batch_size, seq))[:, :, :, -lw:]
        target_signal = target_signal.reshape((n_out, -1, batch_size, seq))[:, :, :, -lw:]

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
        nest.SetStatus(gen_spk_in, params_gen_spk_in)
        nest.SetStatus(gen_rate_target, params_gen_rate_target)

        self.process()
        self.evaluate_curr = evaluate

        duration["sim"] = (
            n_iter * batch_size * duration["sequence"] - duration["total_offset"] - duration["extension_sim"]
        )

        self.prefix_previous = f"{(self.n_iter_sim+1):05d}_{phase_label}"
        self.simulate(duration["sim"], f"{self.prefix_previous}_0_")

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
            self.run_phase("training", eta_train, n_iter_train, True)

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
        duration["task"] = self.n_iter_sim * batch_size * duration["sequence"] + duration["total_offset"]

        gen_spk_final_update.set(dict(spike_times=[duration["task"] + duration["extension_sim"] + 1]))

        self.simulate(duration["final_update"])


training_pipeline = TrainingPipeline()
training_pipeline.run()
training_pipeline.evaluate_final()

if comm.rank != 0:
    exit()

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

if cfg["record_dynamics"]:
    tools.save_recordings("multimeter_rec", duration)
    tools.save_recordings("spike_recorder_in", duration)
    tools.save_recordings("spike_recorder_rec", duration)
    tools.save_recordings("weight_recorder", duration)

tools.verify()

# %% ###########################################################################################################
# Read out post-training weights
# ..............................
# After the training, we can read out the optimized final weights.

tools.save_weights(nrns_inp, nrns_rec, "post_train_inp")
tools.save_weights(nrns_rec, nrns_rec, "post_train_rec")
tools.save_weights(nrns_rec, nrns_out, "post_train_out")

# %% ###########################################################################################################
# Plot results
# ~~~~~~~~~~~~
# Then, we plot a series of plots.

if not cfg["do_plotting"]:
    exit()

# %% ###########################################################################################################
# Read out recorders
# ..................
# We first retrieve the recorded history of the dynamic variables and weights, as well as detected spikes.

df_node_ids = tools.load_data("node_ids")
df_perf = tools.load_data("learning_performance")
df_w_pre_train_inp = tools.load_data("weights_pre_train_inp")
df_w_pre_train_rec = tools.load_data("weights_pre_train_rec")
df_w_pre_train_out = tools.load_data("weights_pre_train_out")
df_w_post_train_inp = tools.load_data("weights_post_train_inp")
df_w_post_train_rec = tools.load_data("weights_post_train_rec")
df_w_post_train_out = tools.load_data("weights_post_train_out")

if cfg["record_dynamics"]:
    df_wr = tools.load_data("weight_recorder_subset")
    df_mm_out = tools.load_data("multimeter_out_subset")
    df_mm_rec = tools.load_data("multimeter_rec_subset")
    df_sr_in = tools.load_data("spike_recorder_in_subset")
    df_sr_rec = tools.load_data("spike_recorder_rec_subset")

# %% ###########################################################################################################
# Plot learning performance
# .........................
# We begin with two plots visualizing the learning performance of the network: the loss and the error, both
# plotted against the iterations.

fig, axs = plt.subplots(2, 1, sharex=True)

for phase in ["validation", "training", "early-stopping", "test"]:
    idc = df_perf.phase == phase
    axs[0].scatter(df_perf.iteration[idc], df_perf.loss[idc], label=phase, marker="x")
    axs[1].scatter(df_perf.iteration[idc], df_perf.error[idc], label=phase, marker="x")

axs[0].set_ylabel(r"$\mathcal{L} = -\sum_{t,k} \pi_k^{*,t} \log \pi_k^t$")
axs[1].set_ylabel("Error")

axs[-1].set_xlabel("Iteration")
axs[-1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
axs[-1].xaxis.get_major_locator().set_params(integer=True)

fig.tight_layout()
fig.savefig(tools.figures_dir / "fig_learning_performance.pdf")

# %% ###########################################################################################################
# Plot weight matrices
# ....................
# If one is not interested in the time course of the weights, it is possible to read out only the initial and
# final weights, which requires less computing time and memory than the weight recorder approach. Here, we plot
# the corresponding weight matrices before and after the optimization.

fig, axs = plt.subplots(3, 2, figsize=(8, 7), sharex="col", sharey="row")

df_w_list = [
    df_w_pre_train_inp,
    df_w_post_train_inp,
    df_w_pre_train_rec,
    df_w_post_train_rec,
    df_w_pre_train_out,
    df_w_post_train_out,
]

w_abs_max = np.max([df.weight.abs().max() for df in df_w_list])
args = dict(
    cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "cmap", ((0.0, tools.colors["blue"]), (0.5, "white"), (1.0, tools.colors["red"]))
    ),
    vmin=-w_abs_max,
    vmax=w_abs_max,
)

for ax, df in zip(axs.flat, df_w_list):
    weight_matrix = df.pivot(index="sender", columns="receiver", values="weight").values
    if weight_matrix.shape[1] != n_rec:
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

cbar = plt.colorbar(cmesh, cax=axs[1, 1].inset_axes([1.1, 0.2, 0.05, 0.8]), label="Weight (pA)")

fig.tight_layout()
fig.savefig(tools.figures_dir / "fig_weight-matrices.pdf")

# %% ###########################################################################################################
# Plot spikes and dynamic variables
# .................................
# This plotting routine shows how to plot all of the recorded dynamic variables and spikes across time. We take
# one snapshot in the first iteration and one snapshot at the end.

if not cfg["record_dynamics"]:
    exit()


def plot_recordable(ax, df, recordable, ylabel, xlims):
    for sender in np.unique(df.sender):
        idc_sender = df.sender == sender
        idc_times = (df.time[idc_sender] > xlims[0]) & (df.time[idc_sender] < xlims[1])
        ax.plot(df.time[idc_sender][idc_times], df[recordable][idc_sender][idc_times])
    ax.set_ylabel(ylabel)
    margin = np.abs(np.max(df[recordable]) - np.min(df[recordable])) * 0.1
    ax.set_ylim(np.min(df[recordable]) - margin, np.max(df[recordable]) + margin)


def plot_spikes(ax, df, ylabel, xlims):
    idc_times = (df.time > xlims[0]) & (df.time < xlims[1])
    senders_subset = df.sender[idc_times]
    times_subset = df.time[idc_times]

    ax.scatter(times_subset, senders_subset, s=0.1, marker="|")
    ax.set_ylabel(ylabel)
    margin = np.abs(np.max(senders_subset) - np.min(senders_subset)) * 0.1
    ax.set_ylim(np.min(senders_subset) - margin, np.max(senders_subset) + margin)


subset = dict(Before=(0, duration["sequence"]), After=(duration["task"] - duration["sequence"], duration["task"]))
for title, xlims in subset.items():
    fig, axs = plt.subplots(9, 1, sharex=True, figsize=(8, 10), gridspec_kw=dict(hspace=0.4, left=0.2))
    fig.suptitle(f"{title} training")

    plot_spikes(axs[0], df_sr_in, r"$z_i$" + "\n", xlims)
    plot_spikes(axs[1], df_sr_rec, r"$z_j$" + "\n", xlims)

    plot_recordable(axs[2], df_mm_rec, "V_m", r"$v_j$" + "\n(mV)", xlims)
    plot_recordable(axs[3], df_mm_rec, "surrogate_gradient", r"$\psi_j$" + "\n", xlims)
    plot_recordable(axs[4], df_mm_rec, "learning_signal", r"$L_j$" + "\n(pA)", xlims)

    plot_recordable(axs[5], df_mm_out, "V_m", r"$v_k$" + "\n(mV)", xlims)
    plot_recordable(axs[6], df_mm_out, "readout_signal", r"$\pi_k$" + "\n", xlims)
    plot_recordable(axs[7], df_mm_out, "target_signal", r"$\pi^*_k$" + "\n", xlims)
    plot_recordable(axs[8], df_mm_out, "error_signal", r"$\pi_k-\pi^*_k$" + "\n", xlims)

    axs[-1].set_xlabel(r"$t$ (ms)")
    axs[-1].set_xlim(*xlims)

    fig.align_ylabels()

fig.savefig(tools.figures_dir / "fig_recordables.pdf")

# %% ###########################################################################################################
# Plot weight time courses
# ........................
# Similarly, we can plot the weight histories. Note that the weight recorder, attached to the synapses, works
# differently than the other recorders. Since synapses only get activated when they transmit a spike, the weight
# recorder only records the weight in those moments. That is why the first weight registrations do not start in
# the first time step and we add the initial weights manually.

id_to_label = df_node_ids.set_index("id")["label"].str.split("_").str[1]
df_wr["label"] = df_wr["sender"].map(id_to_label) + "_" + df_wr["receiver"].map(id_to_label)

weight_matrix_labels = df_wr.label.unique()

fig, axs = plt.subplots(len(weight_matrix_labels), 1, sharex=True, sharey=True, figsize=(5, 5))

w_list = []
for i, (df_w_pre_train, label, short) in enumerate(
    [[df_w_pre_train_inp, "inp_rec", "i"], [df_w_pre_train_rec, "rec_rec", "r"], [df_w_pre_train_out, "rec_out", "o"]]
):
    group = df_wr[df_wr.label == label]
    for sender in np.unique(group.sender):
        for receiver in np.unique(group.receiver):
            df_sub = group[(group.sender == sender) & (group.receiver == receiver)].sort_values(by=["time"])
            initial_weight = df_w_pre_train[
                (df_w_pre_train.sender == sender) & (df_w_pre_train.receiver == receiver)
            ].weight.values
            if len(initial_weight) == 1:
                weights = [initial_weight[0]] + df_sub.weight.tolist()
                times = [0.0] + df_sub.time.tolist()
                axs[i].step(times, weights, c=tools.colors["blue"])
                w_list.append(np.max(np.abs(weights)))

    axs[i].set_ylabel(f"$W^{{\\text{{{short}}}}}$ (pA)")

w_abs_max = np.max(w_list)
axs[-1].set_ylim(-w_abs_max * 1.1, w_abs_max * 1.1)
axs[-1].set_xlabel(r"$t$ (ms)")
axs[-1].set_xlim(0, duration["task"])

fig.align_ylabels()
fig.tight_layout()
fig.savefig(tools.figures_dir / "fig_weight-time-courses.pdf")
