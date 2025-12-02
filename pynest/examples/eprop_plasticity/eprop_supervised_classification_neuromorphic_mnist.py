# -*- coding: utf-8 -*-
#
# eprop_supervised_classification_neuromorphic_mnist.py
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
plasticity mechanism by Bellec et al. [1]_ with additional biological features described in [3]_.

The primary objective of this task is to classify the N-MNIST dataset [2]_, an adaptation of the traditional
MNIST dataset of handwritten digits specifically designed for neuromorphic computing. The N-MNIST dataset
captures changes in pixel intensity through a dynamic vision sensor, converting static images into sequences of
binary events, which we interpret as spike trains. This conversion closely emulates biological neural
processing, making it a fitting challenge for an e-prop-equipped spiking neural network (SNN).

.. image:: eprop_supervised_classification_neuromorphic_mnist.png
   :width: 70 %
   :alt: Schematic of network architecture. Same as Figure 1 in the code.
   :align: center

Learning in the neural network model is achieved by optimizing the connection weights with e-prop plasticity.
This plasticity rule requires a specific network architecture depicted in Figure 1. The neural network model
consists of a recurrent network that receives input from spike generators and projects onto multiple readout
neurons - one for each class. Each input generator is assigned to a pixel of the input image; when an event is
detected in a pixel at time :math:`t`, the corresponding input generator (connected to an input neuron) emits a spike
at that time. Each readout neuron compares the network signal :math:`y_k` with the teacher signal :math:`y_k^*`,
which it receives from a rate generator representing the respective digit class. Unlike conventional neural
network classifiers that may employ softmax functions and cross-entropy loss for classification, this network
model utilizes a mean-squared error loss to evaluate the training error and perform digit classification.

Details on the event-based NEST implementation of e-prop can be found in [3]_.

References
~~~~~~~~~~

.. [1] Bellec G, Scherr F, Subramoney F, Hajek E, Salaj D, Legenstein R, Maass W (2020). A solution to the
       learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11:3625.
       https://doi.org/10.1038/s41467-020-17236-y

.. [2] Orchard, G., Jayawant, A., Cohen, G. K., & Thakor, N. (2015). Converting static image datasets to
       spiking neuromorphic datasets using saccades. Frontiers in neuroscience, 9, 159859.

.. [3] Korcsak-Gorzo A, Stapmanns J, Espinoza Valverde JA, Plesser HE,
       Dahmen D, Bolten M, Van Albada SJ*, Diesmann M*. Event-based
       implementation of eligibility propagation (in preparation)

"""  # pylint: disable=line-too-long # noqa: E501

# %% ###########################################################################################################
# Import libraries
# ~~~~~~~~~~~~~~~~
# We begin by importing all libraries required for the simulation, analysis, and visualization.

from IPython.display import Image
from cycler import cycler
from pathlib import Path
from toolbox import Tools
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpi4py import MPI
import nest
import numpy as np
import requests
import zipfile

# %% ###########################################################################################################
# Schematic of network architecture
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This figure, identical to the one in the description, shows the required network architecture in the center,
# the input and output of the classification task above, and lists of the required NEST device, neuron, and
# synapse models below. The connections that must be established are numbered 1 to 7.

try:
    Image(filename="./eprop_supervised_classification_neuromorphic_mnist.png")
except Exception:
    pass

# %% ###########################################################################################################
# Setup
# ~~~~~

config = dict(
    E_L=-0.1,
    E_L_out=-0.1,
    V_reset=-0.3,
    V_th=0.85,
    batch_size=100,
    c_reg=400.0,
    constrain_weights_dale_in=False,
    constrain_weights_dale_out=False,
    constrain_weights_dale_rec=False,
    constrain_weights_sign_in=False,
    constrain_weights_sign_out=False,
    constrain_weights_sign_rec=False,
    cpus_per_task=1,
    cutoff=100,
    dataset_dir="./",
    do_early_stopping=True,
    eta=5e-6,
    exc_to_inh_ratio=1.0,
    f_target=10.0,
    kappa=0.97,
    kappa_reg=0.99,
    learning_window=300,
    model_nrn_rec="eprop_iaf_psc_delta",
    n_iter_test=10,
    n_iter_train=400,
    n_iter_validate_every=10,
    nodes=1,
    ntasks_per_node=1,
    record_dynamics=False,
    recordings_dir="./recordings",
    recurrent_connectivity=0.1,
    save_weights=True,
    scale_weight_in_rec=0.03,
    scale_weight_out_rec=10.0,
    scale_weight_rec_out=0.006,
    scale_weight_rec_rec=0.02,
    seed=62779435,
    stop_crit=0.05,
    surrogate_gradient="exponential",
    surrogate_gradient_beta=50.0,
    surrogate_gradient_gamma=50.0,
    t_ref=4.0,
    tau_m=10.0,
    tau_m_out=100.0,
)

tools = Tools(config, __file__)
config = tools.config

local_num_threads = config["cpus_per_task"]
total_num_virtual_procs = config["nodes"] * config["ntasks_per_node"] * local_num_threads

# %% ###########################################################################################################
# Initialize random generator
# ...........................
# We seed the numpy random generator, which will generate random initial weights as well as random input and
# output.

rng_seed = config["seed"]  # numpy random seed
np.random.seed(rng_seed)  # fix numpy random seed

# %% ###########################################################################################################
# Define timing of task
# .....................
# The task's temporal structure is then defined, once as time steps and once as durations in milliseconds.
# Even though each sample is processed independently during training, we aggregate predictions and true
# labels across a batch of samples during the evaluation phase. The number of samples in this batch is
# determined by the `batch_size` parameter. This data is then used to assess the neural network's
# performance metrics, such as average accuracy and mean error. Increasing the number of iterations enhances
# learning performance up to the point where overfitting occurs. If early stopping is enabled, the
# classification error is tested in regular intervals and the training stopped as soon as the error selected as
# stop criterion is reached. After training, the performance can be tested over a number of test iterations.

batch_size = config["batch_size"]  # number of instances over which to evaluate the learning performance, 100 for convergence
n_iter_train = config["n_iter_train"]  # number of training iterations, 200 for convergence
n_iter_test = config["n_iter_test"]  # number of iterations for final test
do_early_stopping = config["do_early_stopping"]  # if True, stop training as soon as stop criterion fulfilled
n_iter_validate_every = config["n_iter_validate_every"]  # number of training iterations before validation
n_iter_validate = 1  # number of validation iterations to average over
n_iter_early_stop = 8  # number of iterations to average over to evaluate early stopping condition
stop_crit = config["stop_crit"]  # error value corresponding to stop criterion for early stopping

steps = dict(
    sequence=300,  # time steps of one full sequence
    learning_window=config["learning_window"],  # time steps of window with non-zero learning signals
)

steps.update(
    dict(
        offset_gen=1,  # offset since generator signals start from time step 1
        delay_in_rec=1,  # connection delay between input and recurrent neurons
        extension_sim=1,  # extra time step to close right-open simulation time interval in Simulate()
        final_update=3,  # extra time steps to update all synapses at the end of task
    )
)

steps["delays"] = steps["delay_in_rec"]  # time steps of delays

steps["total_offset"] = steps["offset_gen"] + steps["delays"]  # time steps of total offset

duration = dict(step=1.0)  # ms, temporal resolution of the simulation

duration.update(dict((key, value * duration["step"]) for key, value in steps.items()))  # ms, durations

# %% ###########################################################################################################
# Set up simulation
# .................
# As last step of the setup, we reset the NEST kernel to remove all existing NEST simulation settings and
# objects and set some NEST kernel parameters.

params_setup = dict(
    print_time=False,  # if True, print time progress bar during simulation, set False if run as code cell
    resolution=duration["step"],
    total_num_virtual_procs=total_num_virtual_procs,  # number of virtual processes, set in case of distributed computing
    local_num_threads=local_num_threads,
    overwrite_files=False,  # if True, overwrite existing files
    data_path=str(config["recordings_dir"]),  # path to save data to
    rng_seed=rng_seed,  # seed for random number generator
)

####################

nest.ResetKernel()
nest.set(**params_setup)

comm = MPI.COMM_WORLD

nest.set_verbosity("M_FATAL")

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
    time_max = 336040  # in microseconds, longest recording over training and test set
)

pixels_dict["n_total"] = pixels_dict["n_x"] * pixels_dict["n_y"] * pixels_dict["n_polarity"]  # total number of pixels

n_in = pixels_dict["n_total"]  # number of input neurons = 1196
n_rec = 150  # number of recurrent neurons
n_out = 10  # number of readout neurons

params_nrn_out = dict(
    C_m=1.0,  # pF, membrane capacitance - takes effect only if neurons get current input (here not the case)
    E_L=config["E_L_out"],  # mV, leak / resting membrane potential
    eprop_isi_trace_cutoff=config["cutoff"],  # cutoff of integration of eprop trace between spikes
    I_e=0.0,  # pA, external current input
    tau_m=config["tau_m_out"],  # ms, membrane time constant
    V_m=0.0,  # mV, initial value of the membrane voltage
)

params_nrn_rec = dict(
    beta=config["surrogate_gradient_beta"],  # width scaling of the pseudo-derivative
    C_m=1.0,
    c_reg=config["c_reg"],  # coefficient of firing rate regularization
    E_L=config["E_L"],
    eprop_isi_trace_cutoff=config["cutoff"],
    f_target=config["f_target"],  # spikes/s, target firing rate for firing rate regularization
    gamma=config["surrogate_gradient_gamma"],  # height scaling of the pseudo-derivative
    I_e=0.0,
    kappa=config["kappa"],  # low-pass filter of the eligibility trace
    kappa_reg=config["kappa_reg"],  # low-pass filter of the firing rate for regularization
    surrogate_gradient_function=config["surrogate_gradient"],  # surrogate gradient / pseudo-derivative function
    t_ref=config["t_ref"],  # ms, duration of refractory period
    tau_m=config["tau_m"],
    V_m=0.0,
    V_th=config["V_th"],  # mV, spike threshold membrane voltage
)

if config["model_nrn_rec"] == "eprop_iaf_psc_delta":
   params_nrn_rec["V_reset"] = config["V_reset"]  # mv, reset membrane voltage

####################

# Intermediate parrot neurons required between input spike generators and recurrent neurons,
# since devices cannot establish plastic synapses for technical reasons

gen_spk_in = nest.Create("spike_generator", n_in)
nrns_in = nest.Create("parrot_neuron", n_in)

nrns_rec = nest.Create(config["model_nrn_rec"], n_rec, params_nrn_rec)
nrns_out = nest.Create("eprop_readout", n_out, params_nrn_out)
gen_rate_target = nest.Create("step_rate_generator", n_out)
gen_learning_window = nest.Create("step_rate_generator")

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
    start=duration["offset_gen"] + duration["delay_in_rec"],  # start time of recording
    label="multimeter_rec",
)

params_mm_out = dict(
    interval=duration["step"],
    record_from=["readout_signal", "target_signal"],
    start=duration["total_offset"],
    label="multimeter_out",
)

params_wr = dict(
    senders=nrns_in[:n_record_w] + nrns_rec[:n_record_w],  # limit senders to subsample weights to record
    targets=nrns_rec[:n_record_w] + nrns_out,  # limit targets to subsample weights to record from
    start=duration["total_offset"],
    label="weight_recorder",
)

params_sr_in = dict(
    start=duration["offset_gen"],
    label="spike_recorder_in",
)

params_sr_rec = dict(
    start=duration["offset_gen"],
    label="spike_recorder_rec",
)

for params in [params_mm_rec, params_mm_out, params_wr, params_sr_in, params_sr_rec]:
    params.update(dict(record_to="ascii", precision=16))

####################

if config["record_dynamics"]:
    params_mm_out["record_from"] += ["V_m", "error_signal"]

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
        batch_size=1,
        optimize_each_step=False,  # call optimizer every time step (True) or once per spike (False); both
        # yield same results for gradient descent, False offers speed-up
        Wmin=-100.0,  # pA, minimal limit of the synaptic weights
        Wmax=100.0,  # pA, maximal limit of the synaptic weights
    ),
)

eta_test = 0.0  # learning rate for test phase
eta_train = config["eta"]  # learning rate for training phase

if config["record_dynamics"]:
    # TODO: make sure to only record from existing connections
    nest.SetStatus(wr, params_wr)
    params_common_syn_eprop["weight_recorder"] = wr

params_syn_base = dict(
    synapse_model="eprop_synapse",
    delay=duration["step"],  # ms, dendritic delay
)

params_syn_in = dict(
    synapse_model="eprop_synapse",
    delay=duration["step"],
    weight=nest.random.normal(std=config["scale_weight_in_rec"])
)

params_syn_rec = dict(
    synapse_model="eprop_synapse",
    delay=duration["step"],
    weight=nest.random.normal(std=config["scale_weight_rec_rec"])
)

params_syn_out = dict(
    synapse_model="eprop_synapse",
    delay=duration["step"],
    weight=nest.random.uniform(min=-config["scale_weight_rec_out"], max=config["scale_weight_rec_out"]),
)

params_syn_feedback = dict(
    synapse_model="eprop_learning_signal_connection",
    delay=duration["step"],
    weight=nest.random.normal(std=config["scale_weight_out_rec"]),
)

params_syn_learning_window = dict(
    synapse_model="rate_connection_delayed",
    delay=duration["step"],
    receptor_type=1,  # receptor type over which readout neuron receives learning window signal
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

nest.SetDefaults("eprop_synapse", params_common_syn_eprop)

nest.Connect(gen_spk_in, nrns_in, params_conn_one_to_one, params_syn_static)  # connection 1
nest.Connect(nrns_in, nrns_rec, params_conn_all_to_all, params_syn_in) # connection 2
nest.Connect(nrns_rec, nrns_rec, dict(rule="fixed_indegree", indegree=int(config["recurrent_connectivity"] * n_rec), allow_multapses=False, allow_autapses=False), params_syn_rec) # connection 3
nest.Connect(nrns_rec, nrns_out, params_conn_all_to_all, params_syn_out)  # connection 4
nest.Connect(nrns_out, nrns_rec, params_conn_all_to_all, params_syn_feedback)  # connection 5
nest.Connect(gen_rate_target, nrns_out, params_conn_one_to_one, params_syn_rate_target)  # connection 6
nest.Connect(gen_learning_window, nrns_out, params_conn_all_to_all, params_syn_learning_window)  # connection 7

if config["record_dynamics"]:
    nest.Connect(nrns_in, sr_in, params_conn_all_to_all, params_syn_static)
    nest.Connect(nrns_rec, sr_rec, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_rec, nrns_rec_record, params_conn_all_to_all, params_syn_static)
nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

# TODO check if this function still works after refactoring
tools.constrain_weights(
    nrns_in,
    nrns_rec,
    nrns_out,
    params_syn_base,
    params_common_syn_eprop,
)

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
# The class also includes functionality for random shuffling and grouping of data, ensuring that diverse and
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

    sort_idx = np.lexsort((times, pixel_index)) # sort events first by pixel index, then by time
    times_sorted = times[sort_idx]
    pixels_sorted = pixel_index[sort_idx]

    all_pixel_indices = np.arange(pixels_dict["n_total"])

    # find, for each pixel index, its insertion point in the sorted pixel array, producing the group boundaries
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
            params_gen_rate_target[target_batch[batch_element]]["amplitude_values"][i*batch_size + batch_element] = 1.0

            for n, relative_times in enumerate(input_batch[batch_element]):
                if len(relative_times) > 0:
                    relative_times = np.array(relative_times)
                    spike_times[n].extend(
                        iteration_offset + (i*batch_size + batch_element) * duration["sequence"] + relative_times + duration["offset_gen"]
                    )

    params_gen_spk_in = [dict(spike_times=spk_times) for spk_times in spike_times]

    if duration["sequence"] != duration["learning_window"]:
        params_gen_learning_window = dict(
            amplitude_times=np.hstack(
                [
                    np.array([0.0, duration["sequence"] - duration["learning_window"]])
                    + iteration_offset
                    + batch_element * duration["sequence"]
                    + duration["total_offset"]
                    for batch_element in range(n_iter_curr * batch_size)
                ]
            ),
            amplitude_values=np.tile([0.0, 1.0], n_iter_curr * batch_size),
        )
    else:
        params_gen_learning_window = dict(
            amplitude_times=[duration["total_offset"]],
            amplitude_values=[1.0],
        )

    return params_gen_spk_in, params_gen_rate_target, params_gen_learning_window


save_path = config["dataset_dir"]  # path to save the N-MNIST dataset to
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

nest.Connect(gen_spk_final_update, nrns_in + nrns_rec, "all_to_all", dict(weight=1000.0))

# %% ###########################################################################################################
# Read out pre-training weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Before we begin training, we read out the initial weight matrices so that we can eventually compare them to
# the optimized weights.


def get_weights(pop_pre, pop_post):
    conns = nest.GetConnections(pop_pre, pop_post).get(["source", "target", "weight"])
    conns["senders"] = np.array(conns["source"]) - np.min(conns["source"])
    conns["targets"] = np.array(conns["target"]) - np.min(conns["target"])

    conns["weight_matrix"] = np.zeros((len(pop_post), len(pop_pre)))
    conns["weight_matrix"][conns["targets"], conns["senders"]] = conns["weight"]
    return conns


if config["save_weights"]:
    weights_pre_train = dict(
        in_rec=get_weights(nrns_in, nrns_rec),
        rec_rec=get_weights(nrns_rec, nrns_rec),
        rec_out=get_weights(nrns_rec, nrns_out),
    )

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

    def evaluate(self, events_mm_out):

        readout_signal = events_mm_out.readout_signal
        target_signal = events_mm_out.target_signal
        senders = events_mm_out.sender

        senders_unique = np.unique(senders)

        readout_signal = np.array([readout_signal[senders == i] for i in senders_unique])
        target_signal = np.array([target_signal[senders == i] for i in senders_unique])

        readout_signal = readout_signal.reshape((n_out, -1, batch_size, steps["sequence"]))
        target_signal = target_signal.reshape((n_out, -1, batch_size, steps["sequence"]))

        readout_signal = readout_signal[:, :, :, -steps["learning_window"] :]
        target_signal = target_signal[:, :, :, -steps["learning_window"] :]

        loss = 0.5 * np.mean(np.sum((readout_signal - target_signal) ** 2, axis=3), axis=(0, 2))

        y_prediction = np.argmax(np.mean(readout_signal, axis=3), axis=0)
        y_target = np.argmax(np.mean(target_signal, axis=3), axis=0)
        accuracy = np.mean((y_target == y_prediction), axis=1)
        errors = 1.0 - accuracy
        error = np.mean(errors)

        tools.save_performance(self.n_iter_sim, loss, errors, self.phase_label_previous)
        return error

    def run_phase(self, phase_label, eta, n_iter, loader, evaluate):
        if n_iter == 0:
            return
        tools.set_synapse_defaults(eta)

        params_gen_spk_in, params_gen_rate_target, params_gen_learning_window = get_params_task_input_output(self.n_iter_sim, n_iter, loader)
        nest.SetStatus(gen_spk_in, params_gen_spk_in)
        nest.SetStatus(gen_rate_target, params_gen_rate_target)
        nest.SetStatus(gen_learning_window, params_gen_learning_window)

        self.process()
        self.evaluate_curr = evaluate

        duration["sim"] = n_iter * batch_size * duration["sequence"] - duration["total_offset"] - duration["extension_sim"]

        self.prefix_previous = f"{(self.n_iter_sim+1):05d}_{phase_label}"
        self.simulate(duration["sim"], f"{self.prefix_previous}_0_")

        self.n_iter_sim += n_iter
        self.phase_label_previous = phase_label

    def simulate(self, duration, data_prefix=''):
        nest.data_prefix=data_prefix
        nest.Simulate(duration)

    def run(self):
        if do_early_stopping:
            for self.k_iter in np.arange(0, n_iter_train, n_iter_validate_every):
                self.run_phase("validation", eta_test, n_iter_validate, data_loader_test, True)
                self.run_phase("validation", eta_test, 1, data_loader_test, True)
                if self.k_iter > 0 and self.error < stop_crit:
                    self.run_phase("early-stopping", eta_test, n_iter_early_stop, data_loader_test, True)
                    self.run_phase("early-stopping", eta_test, 1, data_loader_test, True)
                    if self.error < stop_crit:
                        break
                self.run_phase("training", eta_train, n_iter_validate_every, data_loader_train, True)
        else:
            self.run_phase("training", eta_train, n_iter_train, data_loader_train, True)

        self.run_phase("test", eta_test, n_iter_test, data_loader_test, True)

        self.process()

    def process(self):
        data_prefix = f"{self.prefix_previous}_1_" if self.n_iter_sim > 0 else f"{self.n_iter_sim:05d}_offset_0_"
        self.simulate(duration["total_offset"] + duration["extension_sim"], data_prefix)

        error = None

        if comm.rank == 0:
            if self.evaluate_curr:
                events = tools.get_events(self.prefix_previous, save=True)
                error = self.error if events.empty else self.evaluate(events)
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
tools.save_kernel_status(nest.GetKernelStatus())
# training_pipeline.evaluate_final()

n_iter_sim = training_pipeline.n_iter_sim

# %% ###########################################################################################################
# Read out post-training weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# After the training, we can read out the optimized final weights.

if config["save_weights"]:
    weights_post_train = dict(
        in_rec=get_weights(nrns_in, nrns_rec),
        rec_rec=get_weights(nrns_rec, nrns_rec),
        rec_out=get_weights(nrns_rec, nrns_out),
    )

# %% ###########################################################################################################
# Read out recorders
# ~~~~~~~~~~~~~~~~~~
# We can also retrieve the recorded history of the dynamic variables and weights, as well as detected spikes.

if config["save_weights"] and comm.rank == 0:
    tools.save_weights_snapshots(weights_pre_train, weights_post_train)

print("Finished successfully.")
exit()
events_mm_out = tools.get_events("multimeter_out")

if config["record_dynamics"]:
    events_mm_rec = tools.get_events("multimeter_rec")
    events_sr_in = tools.get_events("spike_recorder_in")
    events_sr_rec = tools.get_events("spike_recorder_rec")
    events_wr = tools.get_events("weight_recorder")

results_dict = tools.get_results()
tools.verify()

# %% ###########################################################################################################
# Plot results
# ~~~~~~~~~~~~
# Then, we plot a series of plots.

do_plotting = False  # if True, plot the results

if not do_plotting:
    exit()

colors = dict(
    blue="#2854c5ff",
    red="#e04b40ff",
    green="#25aa2cff",
    gold="#f9c643ff",
    white="#ffffffff",
)

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.prop_cycle": cycler(color=[colors[k] for k in ["blue", "red", "green", "gold"]]),
    }
)

# %% ###########################################################################################################
# Plot learning performance
# .........................
# We begin with two plots visualizing the learning performance of the network: the loss and the error, both
# plotted against the iterations.

fig, axs = plt.subplots(2, 1, sharex=True)
fig.suptitle("Learning performance")

for color, label in zip(colors, set(results_dict["label"])):
    idc = results_dict["label"] == label
    axs[0].scatter(results_dict["iteration"][idc], results_dict["loss"][idc], label=label)
    axs[1].scatter(results_dict["iteration"][idc], results_dict["error"][idc], label=label)

axs[0].set_ylabel(r"$\mathcal{L} = \frac{1}{2} \sum_{t,k} \left( y_k^t -y_k^{*,t}\right)^2$")
axs[1].set_ylabel("error")

axs[-1].set_xlabel("iteration")
axs[-1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
axs[-1].xaxis.get_major_locator().set_params(integer=True)

fig.tight_layout()

# %% ###########################################################################################################
# Plot spikes and dynamic variables
# .................................
# This plotting routine shows how to plot all of the recorded dynamic variables and spikes across time. We take
# one snapshot in the first iteration and one snapshot at the end.


def plot_recordable(ax, events, recordable, ylabel, xlims):
    for sender in np.unique(events["senders"]):
        idc_sender = events["senders"] == sender
        idc_times = (events["times"][idc_sender] > xlims[0]) & (events["times"][idc_sender] < xlims[1])
        ax.plot(events["times"][idc_sender][idc_times], events[recordable][idc_sender][idc_times], lw=0.5)
    ax.set_ylabel(ylabel)
    margin = np.abs(np.max(events[recordable]) - np.min(events[recordable])) * 0.1
    ax.set_ylim(np.min(events[recordable]) - margin, np.max(events[recordable]) + margin)


def plot_spikes(ax, events, ylabel, xlims):
    idc_times = (events["times"] > xlims[0]) & (events["times"] < xlims[1])
    senders_subset = events["senders"][idc_times]
    times_subset = events["times"][idc_times]

    ax.scatter(times_subset, senders_subset, s=0.1)
    ax.set_ylabel(ylabel)
    margin = np.abs(np.max(senders_subset) - np.min(senders_subset)) * 0.1
    ax.set_ylim(np.min(senders_subset) - margin, np.max(senders_subset) + margin)


for title, xlims in zip(
    ["Dynamic variables before training", "Dynamic variables after training"],
    [
        (0, steps["sequence"]),
        ((n_iter_sim - 1) * batch_size * steps["sequence"], n_iter_sim * batch_size * steps["sequence"]),
    ],
):
    fig, axs = plt.subplots(9, 1, sharex=True, figsize=(8, 14), gridspec_kw=dict(hspace=0.4, left=0.2))
    fig.suptitle(title)

    plot_spikes(axs[0], events_sr_in, r"$z_i$" + "\n", xlims)
    plot_spikes(axs[1], events_sr_rec, r"$z_j$" + "\n", xlims)

    plot_recordable(axs[2], events_mm_rec, "V_m", r"$v_j$" + "\n(mV)", xlims)
    plot_recordable(axs[3], events_mm_rec, "surrogate_gradient", r"$\psi_j$" + "\n", xlims)
    plot_recordable(axs[4], events_mm_rec, "learning_signal", r"$L_j$" + "\n(pA)", xlims)

    plot_recordable(axs[5], events_mm_out, "V_m", r"$v_k$" + "\n(mV)", xlims)
    plot_recordable(axs[6], events_mm_out, "target_signal", r"$y^*_k$" + "\n", xlims)
    plot_recordable(axs[7], events_mm_out, "readout_signal", r"$y_k$" + "\n", xlims)
    plot_recordable(axs[8], events_mm_out, "error_signal", r"$y_k-y^*_k$" + "\n", xlims)

    axs[-1].set_xlabel(r"$t$ (ms)")
    axs[-1].set_xlim(*xlims)

    fig.align_ylabels()

# %% ###########################################################################################################
# Plot weight time courses
# ........................
# Similarly, we can plot the weight histories. Note that the weight recorder, attached to the synapses, works
# differently than the other recorders. Since synapses only get activated when they transmit a spike, the weight
# recorder only records the weight in those moments. That is why the first weight registrations do not start in
# the first time step and we add the initial weights manually.


def plot_weight_time_course(ax, events, nrns, label, ylabel):
    sender_label, target_label = label.split("_")
    nrns_senders = nrns[sender_label]
    nrns_targets = nrns[target_label]

    for sender in np.unique(events_wr["senders"]):
        for target in np.unique(events_wr["targets"]):
            if sender in nrns_senders and target in nrns_targets:
                idc_syn = (events["senders"] == sender) & (events["targets"] == target)
                if np.any(idc_syn):
                    idc_syn_pre = (weights_pre_train[label]["source"] == sender) & (
                        weights_pre_train[label]["target"] == target
                    )
                    times = np.concatenate([[0.0], events["times"][idc_syn]])

                    weights = np.concatenate(
                        [np.array(weights_pre_train[label]["weight"])[idc_syn_pre], events["weights"][idc_syn]]
                    )
                    ax.step(times, weights, c=colors["blue"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.6, 0.6)


fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3, 4))
fig.suptitle("Weight time courses")

nrns = {
    "in": nrns_in.tolist(),
    "rec": nrns_rec.tolist(),
    "out": nrns_out.tolist(),
}

plot_weight_time_course(axs[0], events_wr, nrns, "in_rec", r"$W_\text{in}$ (pA)")
plot_weight_time_course(axs[1], events_wr, nrns, "rec_rec", r"$W_\text{rec}$ (pA)")
plot_weight_time_course(axs[2], events_wr, nrns, "rec_out", r"$W_\text{out}$ (pA)")

axs[-1].set_xlabel(r"$t$ (ms)")
axs[-1].set_xlim(0, duration["task"])

fig.align_ylabels()
fig.tight_layout()

# %% ###########################################################################################################
# Plot weight matrices
# ....................
# If one is not interested in the time course of the weights, it is possible to read out only the initial and
# final weights, which requires less computing time and memory than the weight recorder approach. Here, we plot
# the corresponding weight matrices before and after the optimization.

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap", ((0.0, colors["blue"]), (0.5, colors["white"]), (1.0, colors["red"]))
)

fig, axs = plt.subplots(3, 2, sharex="col", sharey="row")
fig.suptitle("Weight matrices")

all_w_extrema = []

for k in weights_pre_train.keys():
    w_pre = weights_pre_train[k]["weight"]
    w_post = weights_post_train[k]["weight"]
    all_w_extrema.append([np.min(w_pre), np.max(w_pre), np.min(w_post), np.max(w_post)])

args = dict(cmap=cmap, vmin=np.min(all_w_extrema), vmax=np.max(all_w_extrema))

for i, weights in zip([0, 1], [weights_pre_train, weights_post_train]):
    axs[0, i].pcolormesh(weights["in_rec"]["weight_matrix"].T, **args)
    axs[1, i].pcolormesh(weights["rec_rec"]["weight_matrix"], **args)
    cmesh = axs[2, i].pcolormesh(weights["rec_out"]["weight_matrix"], **args)

    axs[2, i].set_xlabel("recurrent\nneurons")

axs[0, 0].set_ylabel("input\nneurons")
axs[1, 0].set_ylabel("recurrent\nneurons")
axs[2, 0].set_ylabel("readout\nneurons")
fig.align_ylabels(axs[:, 0])

axs[0, 0].text(0.5, 1.1, "before training", transform=axs[0, 0].transAxes, ha="center")
axs[0, 1].text(0.5, 1.1, "after training", transform=axs[0, 1].transAxes, ha="center")

axs[2, 0].yaxis.get_major_locator().set_params(integer=True)

cbar = plt.colorbar(cmesh, cax=axs[1, 1].inset_axes([1.1, 0.2, 0.05, 0.8]), label="weight (pA)")

fig.tight_layout()

plt.show()
