# -*- coding: utf-8 -*-
#
# eprop_supervised_regression_lemniscate.py
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
Tutorial on learning to generate a lemniscate with e-prop
---------------------------------------------------------

Training a regression model using supervised e-prop plasticity to generate a lemniscate

Description
~~~~~~~~~~~

This script demonstrates supervised learning of a regression task with a recurrent spiking neural network that
is equipped with the eligibility propagation (e-prop) plasticity mechanism by Bellec et al. [1]_ with
additional biological features described in [3]_.

This type of learning is demonstrated at the proof-of-concept task in [1]_. We based this script on their
TensorFlow script given in [2]_ and changed the task as well as the parameters slightly.

In this task, the network learns to generate an arbitrary N-dimensional temporal pattern. Here, the network
learns to reproduce with its overall spiking activity a two-dimensional, roughly one-second-long target signal
which encode the x and y coordinates of a lemniscate.

.. image:: eprop_supervised_regression_lemniscate.png
   :width: 70 %
   :alt: Schematic of network architecture. Same as Figure 1 in the code.
   :align: center

Learning in the neural network model is achieved by optimizing the connection weights with e-prop plasticity.
This plasticity rule requires a specific network architecture depicted in Figure 1. The neural network model
consists of a recurrent network that receives frozen Poisson noise input from spike generators and projects onto two
readout neurons. Each individual readout signal denoted as :math:`y_k` is compared with a corresponding target
signal represented as :math:`y_k^*`. The network's training error is assessed by employing a mean-squared error
loss.

Details on the event-based NEST implementation of e-prop can be found in [3]_.

The development of this task and the hyper-parameter optimization were conducted by Agnes Korcsak-Gorzo and
Charl Linssen, inspired by activities and feedback received at the CapoCaccia Workshop toward Neuromorphic
Intelligence 2023.

References
~~~~~~~~~~

.. [1] Bellec G, Scherr F, Subramoney F, Hajek E, Salaj D, Legenstein R, Maass W (2020). A solution to the
       learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11:3625.
       https://doi.org/10.1038/s41467-020-17236-y

.. [2] https://github.com/IGITUGraz/eligibility_propagation/blob/master/Figure_3_and_S7_e_prop_tutorials/tutorial_pattern_generation.py

.. [3] Korcsak-Gorzo A, Espinoza Valverde JA, Stapmanns J, Plesser HE, Dahmen D,
       Bolten M, van Albada SJ, Diesmann M (2025). Event-driven eligibility
       propagation in large sparse networks: efficiency shaped by biological
       realism. arXiv:2511.21674. https://doi.org/10.48550/arXiv.2511.21674

"""  # pylint: disable=line-too-long # noqa: E501

# %% ###########################################################################################################
# Import libraries
# ~~~~~~~~~~~~~~~~
# We begin by importing all libraries required for the simulation, analysis, and visualization.

import nest
import numpy as np
from mpi4py import MPI
from plotting import Plotter
from toolbox import Tools

# %% ###########################################################################################################
# Setup
# ~~~~~

cfg = dict(
    delete_existing_recordings=False,
    do_plotting=True,
    exc_neuron_fraction=0.5,
    job_cpus_per_task=1,
    job_nodes=1,
    job_ntasks_per_node=1,
    n_iter_train=5,
    record_dynamics=True,
    record_weights=True,
    relative_path_figures_dir="figures",
    relative_path_recordings_dir="recordings",
    save_weights=True,
    seed=1,
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
# the input and output of the pattern generation task above, and lists of the required NEST device, neuron, and
# synapse models below. The connections that must be established are numbered 1 to 6.

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
# Even though each sample is processed independently during training, we aggregate predictions and true
# labels across a batch of samples during the evaluation phase. The number of samples in this batch is
# determined by the `batch_size` parameter. This data is then used to assess the neural network's
# performance metrics, such as average accuracy and mean error. Increasing the number of iterations enhances
# learning performance.

batch_size = 1  # number of instances over which to evaluate the learning performance
n_iter_train = cfg["n_iter_train"]  # number of iterations, 5000 to reach convergence as in the figure

steps = dict(
    sequence=1258,  # time steps of one full sequence
    offset_gen=1,  # offset since generator signals start from time step 1
    delay_inp_rec=1,  # connection delay between input and recurrent neurons
    extension_sim=1,  # extra time step to close right-open simulation time interval in Simulate()
    final_update=3,  # extra time steps to update all synapses at the end of task
)

steps["learning_window"] = steps["sequence"]  # time steps of window with non-zero learning signals
steps["task"] = n_iter_train * batch_size * steps["sequence"]  # time steps of task

steps["delays"] = sum(v for k, v in steps.items() if k.startswith("delay"))  # time steps of delays

steps["total_offset"] = steps["offset_gen"] + steps["delays"]  # time steps of total offset

steps["sim"] = (
    steps["task"] + steps["total_offset"] + steps["extension_sim"] + steps["final_update"]
)  # time steps of simulation

duration = dict(step=1.0)  # ms, temporal resolution of the simulation

duration.update(dict((key, value * duration["step"]) for key, value in steps.items()))  # ms, durations

# %% ###########################################################################################################
# Set up simulation
# .................
# As last step of the setup, we reset the NEST kernel to remove all existing NEST simulation settings and
# objects and set some NEST kernel parameters.

params_setup = dict(
    data_path=str(tools.path_recordings_dir),  # path to save data to
    local_num_threads=local_num_threads,
    overwrite_files=False,  # if True, overwrite existing files
    print_time=False,  # if True, print time progress bar during simulation, set False if run as code cell
    resolution=duration["step"],
    rng_seed=cfg["seed"],  # seed for NEST random generator
    total_num_virtual_procs=total_num_virtual_procs,  # number of virtual processes, set in case of distributed computing
)

####################

nest.ResetKernel()
nest.set(**params_setup)

comm = MPI.COMM_WORLD

# %% ###########################################################################################################
# Create neurons
# ~~~~~~~~~~~~~~
# We proceed by creating a certain number of input, recurrent, and readout neurons and setting their parameters.
# Additionally, we already create an input spike generator and an output target rate generator, which we will
# configure later.

n_in = 100  # number of input neurons
n_rec = 200  # number of recurrent neurons
n_out = 2  # number of readout neurons

params_nrn_out = dict(
    C_m=1.0,  # pF, membrane capacitance - takes effect only if neurons get current input (here not the case)
    E_L=0.0,  # mV, leak / resting membrane potential
    eprop_isi_trace_cutoff=100,  # cutoff of integration of eprop trace between spikes
    I_e=0.0,  # pA, external current input
    tau_m=100.0,  # ms, membrane time constant
    V_m=0.0,  # mV, initial value of the membrane voltage
)

tau_m_mean = 30.0  # ms, mean of membrane time constant distribution

params_nrn_rec = dict(
    beta=33.3,  # width scaling of the pseudo-derivative
    adapt_tau=2000.0,  # ms, time constant of adaptive threshold
    C_m=250.0,
    c_reg=0.12,  # coefficient of firing rate regularization
    E_L=0.0,
    eprop_isi_trace_cutoff=100,
    f_target=20.0,  # spikes/s, target firing rate for firing rate regularization
    gamma=10.0,  # height scaling of the pseudo-derivative
    I_e=0.0,
    kappa=0.99,  # low-pass filter of the eligibility trace
    kappa_reg=0.99,  # low-pass filter of the firing rate for regularization
    surrogate_gradient_function="piecewise_linear",  # surrogate gradient / pseudo-derivative function
    t_ref=0.0,  # ms, duration of refractory period
    tau_m=nest.random.normal(mean=tau_m_mean, std=2.0),
    V_m=0.0,
    V_th=0.03,  # mV, spike threshold membrane voltage
    flush_event_send_interval=duration["sequence"],
)

params_nrn_rec["adapt_beta"] = (
    1.7 * (1.0 - np.exp(-1 / params_nrn_rec["adapt_tau"])) / (1.0 - np.exp(-1.0 / tau_m_mean))
)  # prefactor of adaptive threshold

####################

# Intermediate parrot neurons required between input spike generators and recurrent neurons,
# since devices cannot establish plastic synapses for technical reasons

gen_spk_in = nest.Create("spike_generator", n_in)
nrns_inp = nest.Create("parrot_neuron", n_in)

nrns_rec = nest.Create("eprop_iaf_adapt", n_rec, params_nrn_rec)
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
    record_from=[
        "V_m",
        "surrogate_gradient",
        "learning_signal",
        "V_th_adapt",
        "adaptation",
    ],  # dynamic variables to record
    start=duration["offset_gen"] + duration["delay_inp_rec"],  # start time of recording
    stop=duration["offset_gen"] + duration["delay_inp_rec"] + duration["task"],  # stop time of recording
    label="multimeter_rec",
    record_to="ascii",
    precision=16,
)

params_mm_out = dict(
    interval=duration["step"],
    record_from=["readout_signal", "target_signal"],
    start=duration["total_offset"],
    stop=duration["total_offset"] + duration["task"],
    label="multimeter_out",
    record_to="ascii",
    precision=16,
)

params_wr = dict(
    start=duration["total_offset"],
    stop=duration["sim"],
    label="weight_recorder",
    record_to="ascii",
    precision=16,
)

params_sr_in = dict(
    start=duration["offset_gen"],
    stop=duration["total_offset"] + duration["task"],
    label="spike_recorder_in",
    record_to="ascii",
    precision=16,
)

params_sr_rec = dict(
    start=duration["offset_gen"],
    stop=duration["total_offset"] + duration["task"],
    label="spike_recorder_rec",
    record_to="ascii",
    precision=16,
)

####################

if cfg["record_dynamics"]:
    params_mm_out["record_from"] += ["V_m", "error_signal"]

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

gen_spk_final_update = nest.Create(
    "spike_generator",
    1,
    dict(spike_times=[duration["task"] + duration["total_offset"] + duration["extension_sim"] + 1.0]),
)

# %% ###########################################################################################################
# Create connections
# ~~~~~~~~~~~~~~~~~~
# Now, we define the connectivity and set up the synaptic parameters, with the synaptic weights drawn from
# random distributions. After these preparations, we establish the enumerated connections of the core network,
# as well as additional connections to the recorders.

params_conn_all_to_all = dict(rule="all_to_all", allow_autapses=False)
params_conn_one_to_one = dict(rule="one_to_one")

dtype_weights = np.float32  # data type of weights - for reproducing TF results set to np.float32
weights_inp_rec = np.array(np.random.randn(n_in, n_rec).T / np.sqrt(n_in), dtype=dtype_weights) * 0.01
weights_rec_rec = np.array(np.random.randn(n_rec, n_rec).T / np.sqrt(n_rec), dtype=dtype_weights) * 0.01
np.fill_diagonal(weights_rec_rec, 0.0)  # since no autapses set corresponding weights to zero
weights_rec_out = np.array(np.random.randn(n_rec, n_out).T / np.sqrt(n_rec), dtype=dtype_weights) * 0.01
weights_out_rec = np.array(np.random.randn(n_rec, n_out) / np.sqrt(n_rec), dtype=dtype_weights) * 0.01

params_common_syn_eprop = dict(
    optimizer=dict(
        type="adam",  # algorithm to optimize the weights
        batch_size=1,
        beta_1=0.9,  # exponential decay rate for 1st moment estimate of Adam optimizer
        beta_2=0.999,  # exponential decay rate for 2nd moment raw estimate of Adam optimizer
        epsilon=1e-8,  # small numerical stabilization constant of Adam optimizer
        eta=5e-7,  # learning rate
        optimize_each_step=False,  # call optimizer every time step (True) or once per spike (False); both
        # yield same results for gradient descent, False offers speed-up
        Wmin=-100.0,  # pA, minimal limit of the synaptic weights
        Wmax=100.0,  # pA, maximal limit of the synaptic weights
    ),
)

plastic_synapse_model = "eprop_synapse"

params_syn_in = dict(
    synapse_model=plastic_synapse_model,
    delay=duration["step"],
    weight=weights_inp_rec,  # pA, initial values for the synaptic weights
)

params_syn_rec = dict(
    synapse_model=plastic_synapse_model,
    delay=duration["step"],
    weight=weights_rec_rec,
)

params_syn_out = dict(
    synapse_model=plastic_synapse_model,
    delay=duration["step"],
    weight=weights_rec_out,
)

params_syn_feedback = dict(
    synapse_model="eprop_learning_signal_connection",
    delay=duration["step"],
    weight=weights_out_rec,
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

params_init_optimizer = dict(
    optimizer=dict(
        m=0.0,  # initial 1st moment estimate m of Adam optimizer
        v=0.0,  # initial 2nd moment raw estimate v of Adam optimizer
    )
)

####################

nest.SetDefaults(plastic_synapse_model, params_common_syn_eprop)

nest.Connect(gen_spk_in, nrns_inp, params_conn_one_to_one, params_syn_static)  # connection 1
nest.Connect(nrns_inp, nrns_rec, params_conn_all_to_all, params_syn_in)  # connection 2
nest.Connect(nrns_rec, nrns_rec, params_conn_all_to_all, params_syn_rec)  # connection 3
nest.Connect(nrns_rec, nrns_out, params_conn_all_to_all, params_syn_out)  # connection 4
nest.Connect(nrns_out, nrns_rec, params_conn_all_to_all, params_syn_feedback)  # connection 5
nest.Connect(gen_rate_target, nrns_out, params_conn_one_to_one, params_syn_rate_target)  # connection 6
nest.Connect(gen_learning_window, nrns_out, params_conn_all_to_all, params_syn_learning_window)  # connection 7
nest.Connect(gen_spk_final_update, nrns_inp + nrns_rec, "all_to_all", dict(weight=1000.0))
nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

if cfg["record_dynamics"]:
    nest.Connect(nrns_inp, sr_in, params_conn_all_to_all, params_syn_static)
    nest.Connect(nrns_rec, sr_rec, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_rec, nrns_rec_record, params_conn_all_to_all, params_syn_static)

if cfg["record_weights"]:
    tools.configure_weight_recorder_connections(wr, nrns_inp, nrns_rec, nrns_out, n_record_w)
    nest.SetDefaults(plastic_synapse_model, dict(weight_recorder=wr))

# After creating the connections, we can individually initialize the optimizer's
# dynamic variables for single synapses (here exemplarily for two connections).

nest.GetConnections(nrns_rec[0], nrns_rec[1:3]).set([params_init_optimizer] * 2)

tools.constrain_weights(nrns_inp, nrns_rec, params_syn_in, "inp")
tools.constrain_weights(nrns_rec, nrns_rec, params_syn_rec, "rec")
tools.constrain_weights(nrns_rec, nrns_out, params_syn_out, "out")

# %% ###########################################################################################################
# Create input
# ~~~~~~~~~~~~
# We generate some frozen Poisson spike noise of a fixed rate that is repeated in each iteration and feed these
# spike times to the previously created input spike generator. The network will use these spike times as a
# temporal backbone for encoding the target signal into its recurrent spiking activity.

input_spike_prob = 0.05  # spike probability of frozen input noise
dtype_in_spks = np.float32  # data type of input spikes - for reproducing TF results set to np.float32

input_spike_bools = (np.random.rand(steps["sequence"], n_in) < input_spike_prob).swapaxes(0, 1)

sequence_starts = np.arange(0.0, duration["task"], duration["sequence"]) + duration["offset_gen"]
params_gen_spk_in = []
for input_spike_bool in input_spike_bools:
    input_spike_times = np.arange(0.0, duration["sequence"], duration["step"])[input_spike_bool]
    input_spike_times_all = [input_spike_times + start for start in sequence_starts]
    params_gen_spk_in.append(dict(spike_times=np.hstack(input_spike_times_all).astype(dtype_in_spks)))

####################

gen_spk_in.set(params_gen_spk_in)

# %% ###########################################################################################################
# Create output
# ~~~~~~~~~~~~~
# Then, we load the x and y values of an image of a lemniscate and construct a roughly
# one-second long target signal from it. This signal, like the input, is repeated for all iterations and fed
# into the rate generator that was previously created.

target_signal_list = [
    np.sin(np.linspace(0.0, 2.0 * np.pi, steps["sequence"])),
    np.sin(np.linspace(0.0, 4.0 * np.pi, steps["sequence"])),
]

params_gen_rate_target = []

for target_signal in target_signal_list:
    params_gen_rate_target.append(
        dict(
            amplitude_times=np.arange(0.0, duration["task"], duration["step"]) + duration["total_offset"],
            amplitude_values=np.tile(target_signal, n_iter_train * batch_size),
        )
    )

####################

gen_rate_target.set(params_gen_rate_target)

# %% ###########################################################################################################
# Create learning window
# ~~~~~~~~~~~~~~~~~~~~~~
# Custom learning windows, in which the network learns, can be defined with an additional signal. The error
# signal is internally multiplied with this learning window signal. Passing a learning window signal of value 1
# opens the learning window while passing a value of 0 closes it.

params_gen_learning_window = dict(
    amplitude_times=[duration["total_offset"]],
    amplitude_values=[1.0],
)

####################

gen_learning_window.set(params_gen_learning_window)

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
# Simulate
# ~~~~~~~~
# We train the network by simulating for a set simulation time, determined by the number of iterations and the
# batch size and the length of one sequence.

nest.Simulate(duration["sim"])

if comm.rank != 0:
    exit()

# %% ###########################################################################################################
# Evaluate training error
# ~~~~~~~~~~~~~~~~~~~~~~~
# We evaluate the network's training error by calculating a loss - in this case, the mean squared error between
# the integrated recurrent network activity and the target rate.

senders, readout_signal, target_signal = tools.get_events()

readout_signal = readout_signal.reshape((n_out, n_iter_train, batch_size, steps["sequence"]))
target_signal = target_signal.reshape((n_out, n_iter_train, batch_size, steps["sequence"]))

loss = 0.5 * np.mean(np.sum((readout_signal - target_signal) ** 2, axis=3), axis=(0, 2))

tools.save_performance(0, loss, phase_label="training")

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
        include_plot_pattern=True,
    ).plot_all()
