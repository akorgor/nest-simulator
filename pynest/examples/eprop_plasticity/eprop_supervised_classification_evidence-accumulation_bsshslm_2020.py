# -*- coding: utf-8 -*-
#
# eprop_supervised_classification_evidence-accumulation_bsshslm_2020.py
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
Tutorial on learning to accumulate evidence with e-prop after Bellec et al. (2020)
----------------------------------------------------------------------------------

Training a classification model using supervised e-prop plasticity to accumulate evidence.

Description
~~~~~~~~~~~

This script demonstrates supervised learning of a classification task with the eligibility propagation (e-prop)
plasticity mechanism by Bellec et al. [1]_.

This type of learning is demonstrated at the proof-of-concept task in [1]_. We based this script on their
TensorFlow script given in [2]_.

The task, a so-called evidence accumulation task, is inspired by behavioral tasks, where a lab animal (e.g., a
mouse) runs along a track, gets cues on the left and right, and has to decide at the end of the track between
taking a left and a right turn of which one is correct. After a number of iterations, the animal is able to
infer the underlying rationale of the task. Here, the solution is to turn to the side in which more cues were
presented.

.. image:: eprop_supervised_classification_evidence-accumulation_bsshslm_2020.png
   :width: 70 %
   :alt: Schematic of network architecture. Same as Figure 1 in the code.
   :align: center

Learning in the neural network model is achieved by optimizing the connection weights with e-prop plasticity.
This plasticity rule requires a specific network architecture depicted in Figure 1. The neural network model
consists of a recurrent network that receives input from spike generators and projects onto two readout
neurons - one for the left and one for the right turn at the end. The input neuron population consists of four
groups: one group providing background noise of a specific rate for some base activity throughout the
experiment, one group providing the input spikes of the left cues and one group providing them for the right
cues, and a last group defining the recall window, in which the network has to decide. The readout neuron
compares the network signal :math:`\pi_k` with the target signal :math:`\pi_k^*`, which it receives from
a rate generator. Since the decision is at the end and all the cues are relevant, the network has to keep the
cues in memory. Additional adaptive neurons in the network enable this memory. The network's training error is
assessed by employing a cross-entropy error loss.

Details on the event-based NEST implementation of e-prop can be found in [3]_.

References
~~~~~~~~~~

.. [1] Bellec G, Scherr F, Subramoney F, Hajek E, Salaj D, Legenstein R, Maass W (2020). A solution to the
       learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11:3625.
       https://doi.org/10.1038/s41467-020-17236-y

.. [2] https://github.com/IGITUGraz/eligibility_propagation/blob/master/Figure_3_and_S7_e_prop_tutorials/tutorial_evidence_accumulation_with_alif.py

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
    batch_size=2,
    delete_existing_recordings=False,
    do_early_stopping=False,
    do_plotting=True,
    eta=5e-3,
    exc_neuron_fraction=0.5,
    job_cpus_per_task=1,
    job_nodes=1,
    job_ntasks_per_node=1,
    loss="cross_entropy",
    n_iter_test=1,
    n_iter_train=5,
    n_iter_train_chunk=5,
    n_iter_validate_every=10,
    perturbation=False,
    record_dynamics=True,
    record_weights=True,
    record_n_seq=1,
    relative_path_figures_dir="figures",
    relative_path_recordings_dir="recordings",
    reset_neurons=True,
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
# The original number of iterations requires distributed computing. Increasing the number of iterations
# enhances learning performance up to the point where overfitting occurs. If early stopping is enabled, the
# classification error is tested in regular intervals and the training stopped as soon as the error selected as
# stop criterion is reached. After training, the performance can be tested over a number of test iterations.

batch_size = cfg["batch_size"]  # batch size, 64 in reference [2], 32 in the README to reference [2]
n_iter_train = cfg["n_iter_train"]  # number of training iterations, 2000 in reference [2]
n_iter_train_chunk = cfg["n_iter_train_chunk"]  # chunk training iterations to reduce memory usage
n_iter_test = cfg["n_iter_test"]  # number of iterations for final test
do_early_stopping = cfg["do_early_stopping"]  # if True, stop training as soon as stop criterion fulfilled
n_iter_validate_every = cfg["n_iter_validate_every"]  # number of training iterations before validation
n_iter_validate = 1  # number of validation iterations to average over
n_iter_early_stop = 8  # number of iterations to average over to evaluate early stopping condition
stop_crit = 0.07  # error value corresponding to stop criterion for early stopping

input = dict(
    n_symbols=4,  # number of input populations, e.g. 4 = left, right, recall, noise
    n_cues=7,  # number of cues given before decision
    prob_group=0.3,  # probability with which one input group is present
    spike_prob=0.04,  # spike probability of frozen input noise
)

steps = dict(
    cue=100,  # time steps in one cue presentation
    spacing=50,  # time steps of break between two cues
    bg_noise=1050,  # time steps of background noise
    recall=150,  # time steps of recall
    offset_gen=1,  # offset since generator signals start from time step 1
    delay_inp_rec=1,  # connection delay between input and recurrent neurons
    delay_rec_out=1,  # connection delay between recurrent and output neurons
    delay_out_norm=1,  # connection delay between output neurons for normalization
    extension_sim=1,  # extra time step to close right-open simulation time interval in Simulate()
    final_update=3,  # extra time steps to update all synapses at the end of task
)

steps["cues"] = input["n_cues"] * (steps["cue"] + steps["spacing"])  # time steps of all cues
steps["sequence"] = steps["cues"] + steps["bg_noise"] + steps["recall"]  # time steps of one full sequence
steps["learning_window"] = steps["recall"]  # time steps of window with non-zero learning signals

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
# configure later. Within the recurrent network, alongside a population of regular neurons, we introduce a
# population of adaptive neurons, to enhance the network's memory retention.

n_in = 40  # number of input neurons
n_ad = 50  # number of adaptive neurons
n_reg = 50  # number of regular neurons
n_rec = n_ad + n_reg  # number of recurrent neurons
n_out = 2  # number of readout neurons

params_nrn_out = dict(
    C_m=1.0,  # pF, membrane capacitance - takes effect only if neurons get current input (here not the case)
    E_L=0.0,  # mV, leak / resting membrane potential
    I_e=0.0,  # pA, external current input
    loss=cfg["loss"],  # loss function
    regular_spike_arrival=False,  # If True, input spikes arrive at end of time step, if False at beginning
    tau_m=20.0,  # ms, membrane time constant
    V_m=0.0,  # mV, initial value of the membrane voltage
)

params_nrn_reg = dict(
    C_m=1.0,
    c_reg=300.0,  # coefficient of firing rate regularization - 2*learning_window*(TF c_reg) for technical reasons
    E_L=0.0,
    f_target=10.0,  # spikes/s, target firing rate for firing rate regularization
    I_e=0.0,
    regular_spike_arrival=True,
    surrogate_gradient_function="piecewise_linear",  # surrogate gradient / pseudo-derivative function
    surrogate_gradient_height=0.3,  # height scaling of the pseudo-derivative
    surrogate_gradient_width=1.0,  # width scaling of the pseudo-derivative
    t_ref=5.0,  # ms, duration of refractory period
    tau_m=20.0,
    V_m=0.0,
    V_th=0.6,  # mV, spike threshold membrane voltage
    flush_event_send_interval=duration["sequence"],
)

# factors from the original pseudo-derivative definition are incorporated into the parameters
params_nrn_reg["surrogate_gradient_height"] /= params_nrn_reg["V_th"]
params_nrn_reg["surrogate_gradient_width"] *= np.abs(params_nrn_reg["V_th"])  # prefactor is inside abs in the original definition

params_nrn_ad = dict(
    adapt_tau=2000.0,  # ms, time constant of adaptive threshold
    adaptation=0.0,  # initial value of the spike threshold adaptation
    C_m=1.0,
    c_reg=300.0,
    E_L=0.0,
    f_target=10.0,
    I_e=0.0,
    regular_spike_arrival=True,
    surrogate_gradient_function="piecewise_linear",
    surrogate_gradient_height=0.3,
    surrogate_gradient_width=1.0,
    t_ref=5.0,
    tau_m=20.0,
    V_m=0.0,
    V_th=0.6,
)

params_nrn_ad["surrogate_gradient_height"] /= params_nrn_ad["V_th"]
params_nrn_ad["surrogate_gradient_width"] *= np.abs(params_nrn_ad["V_th"])

params_nrn_ad["adapt_beta"] = 1.7 * (
    (1.0 - np.exp(-duration["step"] / params_nrn_ad["adapt_tau"]))
    / (1.0 - np.exp(-duration["step"] / params_nrn_ad["tau_m"]))
)  # prefactor of adaptive threshold

####################

# Intermediate parrot neurons required between input spike generators and recurrent neurons,
# since devices cannot establish plastic synapses for technical reasons

gen_spk_in = nest.Create("spike_generator", n_in)
nrns_inp = nest.Create("parrot_neuron", n_in)

# The suffix _bsshslm_2020 follows the NEST convention to indicate in the model name the paper
# that introduced it by the first letter of the authors' last names and the publication year.

nrns_reg = nest.Create("eprop_iaf_bsshslm_2020", n_reg, params_nrn_reg)
nrns_ad = nest.Create("eprop_iaf_adapt_bsshslm_2020", n_ad, params_nrn_ad)
nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
gen_rate_target = nest.Create("step_rate_generator", n_out)

nrns_rec = nrns_reg + nrns_ad

# %% ###########################################################################################################
# Create recorders
# ~~~~~~~~~~~~~~~~
# We also create recorders, which, while not required for the training, will allow us to track various dynamic
# variables of the neurons, spikes, and changes in synaptic weights. To save computing time and memory, the
# recorders, the recorded variables, neurons, and synapses can be limited to the ones relevant to the
# experiment, and the recording interval can be increased (see the documentation on the specific recorders). By
# default, recordings are stored in memory but can also be written to file.

n_record = 1  # number of neurons per type to record dynamic variables from - this script requires n_record >= 1
n_record_w = 5  # number of senders and targets to record weights from - this script requires n_record_w >=1

if n_record == 0 or n_record_w == 0:
    raise ValueError("n_record and n_record_w >= 1 required")

params_mm_learning = dict(
    interval=duration["step"],
    label="multimeter_learning",
    precision=16,
    record_from=["readout_signal", "target_signal"],
    record_to="ascii",
    start=duration["total_offset"],
)

params_mm_out = dict(
    interval=duration["step"],
    label="multimeter_out",
    precision=16,
    record_from=["readout_signal", "target_signal", "V_m", "readout_signal_unnorm", "error_signal"],
    record_to="ascii",
    start=0.0,
    stop=cfg["record_n_seq"]*duration["sequence"],
)

params_mm_reg = dict(
    interval=duration["step"],  # interval between two recorded time points
    label="multimeter_reg",
    precision=16,
    record_from=["V_m", "surrogate_gradient", "learning_signal"],  # dynamic variables to record
    record_to="ascii",
    start=0.0,  # start time of recording
    stop=cfg["record_n_seq"]*duration["sequence"],  # stop time of recording
)

params_mm_ad = dict(
    interval=duration["step"],
    label="multimeter_ad",
    precision=16,
    record_from=params_mm_reg["record_from"] + ["V_th_adapt", "adaptation"],
    record_to="ascii",
    start=0.0,  # start time of recording
    stop=cfg["record_n_seq"]*duration["sequence"],  # stop time of recording
)

params_sr_in = dict(
    label="spike_recorder_in",
    precision=16,
    record_to="ascii",
    start=0.0,
    stop=cfg["record_n_seq"]*duration["sequence"],
)

params_sr_reg = dict(
    label="spike_recorder_reg",
    precision=16,
    record_to="ascii",
    start=0.0,
    stop=cfg["record_n_seq"]*duration["sequence"],
)

params_sr_ad = dict(
    label="spike_recorder_ad",
    precision=16,
    record_to="ascii",
    start=0.0,
    stop=cfg["record_n_seq"]*duration["sequence"],
)

params_wr = dict(
    label="weight_recorder",
    precision=16,
    record_to="ascii",
    start=0.0,
    stop=cfg["record_n_seq"]*duration["sequence"],
)

####################

mm_learning = nest.Create("multimeter", params_mm_learning)

if cfg["record_dynamics"]:
    mm_out = nest.Create("multimeter", params_mm_out)
    mm_reg = nest.Create("multimeter", params_mm_reg)
    mm_ad = nest.Create("multimeter", params_mm_ad)
    sr_in = nest.Create("spike_recorder", params_sr_in)
    sr_reg = nest.Create("spike_recorder", params_sr_reg)
    sr_ad = nest.Create("spike_recorder", params_sr_ad)
    dynamics_recorders = [mm_out, mm_reg, mm_ad, sr_in, sr_reg, sr_ad]

if cfg["record_weights"]:
    wr = nest.Create("weight_recorder", params_wr)

nrns_reg_record = nrns_reg[:n_record]
nrns_ad_record = nrns_ad[:n_record]

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


def calculate_glorot_dist(fan_in, fan_out):
    glorot_scale = 1.0 / max(1.0, (fan_in + fan_out) / 2.0)
    glorot_limit = np.sqrt(3.0 * glorot_scale)
    glorot_distribution = np.random.uniform(low=-glorot_limit, high=glorot_limit, size=(fan_in, fan_out))
    return glorot_distribution


dtype_weights = np.float32  # data type of weights - for reproducing TF results set to np.float32
weights_inp_rec = np.array(np.random.randn(n_in, n_rec).T / np.sqrt(n_in), dtype=dtype_weights)
weights_rec_rec = np.array(np.random.randn(n_rec, n_rec).T / np.sqrt(n_rec), dtype=dtype_weights)
np.fill_diagonal(weights_rec_rec, 0.0)  # since no autapses set corresponding weights to zero
weights_rec_out = np.array(calculate_glorot_dist(n_rec, n_out).T, dtype=dtype_weights)
weights_out_rec = np.array(np.random.randn(n_rec, n_out), dtype=dtype_weights)

params_common_syn_eprop = dict(
    optimizer=dict(
        type="adam",  # algorithm to optimize the weights
        batch_size=batch_size,
        beta_1=0.9,  # exponential decay rate for 1st moment estimate of Adam optimizer
        beta_2=0.999,  # exponential decay rate for 2nd moment raw estimate of Adam optimizer
        epsilon=1e-8,  # small numerical stabilization constant of Adam optimizer
        Wmin=-100.0,  # pA, minimal limit of the synaptic weights
        Wmax=100.0,  # pA, maximal limit of the synaptic weights
    ),
    average_gradient=True,  # if True, average the gradient over the learning window
)

eta_test = 0.0  # learning rate for test phase
eta_train = cfg["eta"]  # learning rate for training phase

plastic_synapse_model = "eprop_synapse_bsshslm_2020"

params_syn_in = dict(
    synapse_model=plastic_synapse_model,
    delay=duration["step"],
    tau_m_readout=params_nrn_out["tau_m"],
    weight=weights_inp_rec,  # pA, initial values for the synaptic weights
)

params_syn_rec = dict(
    synapse_model=plastic_synapse_model,
    delay=duration["step"],
    tau_m_readout=params_nrn_out["tau_m"],
    weight=weights_rec_rec,
)

params_syn_out = dict(
    synapse_model=plastic_synapse_model,
    delay=duration["step"],
    tau_m_readout=params_nrn_out["tau_m"],
    weight=weights_rec_out,
)

params_syn_feedback = dict(
    synapse_model="eprop_learning_signal_connection_bsshslm_2020",
    delay=duration["step"],
    weight=weights_out_rec,
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
nest.Connect(nrns_out, nrns_out, params_conn_all_to_all, params_syn_out_out)  # connection 7
nest.Connect(gen_spk_final_update, nrns_inp + nrns_rec, "all_to_all", dict(weight=1000.0))
nest.Connect(mm_learning, nrns_out, params_conn_all_to_all, params_syn_static)

if cfg["record_dynamics"]:
    nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_reg, nrns_reg_record, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_ad, nrns_ad_record, params_conn_all_to_all, params_syn_static)
    nest.Connect(nrns_inp, sr_in, params_conn_all_to_all, params_syn_static)
    nest.Connect(nrns_reg, sr_reg, params_conn_all_to_all, params_syn_static)
    nest.Connect(nrns_ad, sr_ad, params_conn_all_to_all, params_syn_static)

if cfg["record_weights"]:
    tools.configure_weight_recorder_connections(wr, nrns_inp, nrns_rec, nrns_out, n_record_w)
    nest.SetDefaults(plastic_synapse_model, dict(weight_recorder=wr))

# After creating the connections, we can individually initialize the optimizer's
# dynamic variables for single synapses (here exemplarily for two connections).

nest.GetConnections(nrns_rec[0], nrns_rec[1:3]).set([params_init_optimizer] * 2)

tools.constrain_weights(nrns_inp, nrns_rec, params_syn_in, "inp")
tools.constrain_weights(nrns_rec, nrns_rec, params_syn_rec, "rec")
tools.constrain_weights(nrns_rec, nrns_out, params_syn_out, "out")

if cfg["perturbation"]:
    sg_perturb = nest.Create("spike_generator", dict(spike_times=[duration["sequence"]*batch_size*2  + duration["cues"] + duration["bg_noise"] + 20.0]))
    conns_dict = nest.GetConnections(nrns_rec[0], nrns_rec+nrns_out).get(["target", "weight"])
    conn_target = np.array(conns_dict["target"])
    conn_weight= np.array(conns_dict["weight"])
    idc = np.argsort(conn_target)
    nest.Connect([sg_perturb.get("global_id")]*len(idc), conn_target[idc], conn_spec="one_to_one", syn_spec=dict(weight=conn_weight[idc]))

# %% ###########################################################################################################
# Create input and output
# ~~~~~~~~~~~~~~~~~~~~~~~
# We generate the input as four neuron populations, two producing the left and right cues, respectively, one the
# recall signal and one the background input throughout the task. The sequence of cues is drawn with a
# probability that favors one side. For each such sequence, the favored side, the solution or target, is
# assigned randomly to the left or right.


def generate_evidence_accumulation_input_output(batch_size, n_in, steps, input):
    n_pop_nrn = n_in // input["n_symbols"]

    prob_choices = np.array([input["prob_group"], 1 - input["prob_group"]], dtype=np.float32)
    idx = np.random.choice([0, 1], batch_size)
    probs = np.zeros((batch_size, 2), dtype=np.float32)
    probs[:, 0] = prob_choices[idx]
    probs[:, 1] = prob_choices[1 - idx]

    batched_cues = np.zeros((batch_size, input["n_cues"]), dtype=int)
    for b_idx in range(batch_size):
        batched_cues[b_idx, :] = np.random.choice([0, 1], input["n_cues"], p=probs[b_idx])

    input_spike_probs = np.zeros((batch_size, steps["sequence"], n_in))

    for b_idx in range(batch_size):
        for c_idx in range(input["n_cues"]):
            cue = batched_cues[b_idx, c_idx]

            step_start = c_idx * (steps["cue"] + steps["spacing"]) + steps["spacing"]
            step_stop = step_start + steps["cue"]

            pop_nrn_start = cue * n_pop_nrn
            pop_nrn_stop = pop_nrn_start + n_pop_nrn

            input_spike_probs[b_idx, step_start:step_stop, pop_nrn_start:pop_nrn_stop] = input["spike_prob"]

    input_spike_probs[:, -steps["recall"] :, 2 * n_pop_nrn : 3 * n_pop_nrn] = input["spike_prob"]
    input_spike_probs[:, :, 3 * n_pop_nrn :] = input["spike_prob"] / 4.0
    input_spike_bools = input_spike_probs > np.random.rand(input_spike_probs.size).reshape(input_spike_probs.shape)
    input_spike_bools[:, 0, :] = 0  # remove spikes in 0th time step of every sequence for technical reasons

    target_cues = np.zeros(batch_size, dtype=int)
    target_cues[:] = np.sum(batched_cues, axis=1) > int(input["n_cues"] / 2)

    return input_spike_bools, target_cues


def get_params_task_input_output(n_iter_interval, n_iter_curr):
    iteration_offset = n_iter_interval * batch_size * duration["sequence"]
    dtype_in_spks = np.float32  # data type of input spikes - for reproducing TF results set to np.float32

    input_spike_bools_arr_list = []
    target_cues_list = []
    for _ in range(n_iter_curr):
        input_spike_bools, target_cues = generate_evidence_accumulation_input_output(batch_size, n_in, steps, input)
        input_spike_bools_arr_list.append(input_spike_bools.reshape(batch_size * steps["sequence"], n_in))
        target_cues_list.append(target_cues)

    input_spike_bools_arr = np.vstack(input_spike_bools_arr_list)
    target_cues_arr = np.hstack(target_cues_list)
    timeline_task = (
        np.arange(0.0, n_iter_curr * batch_size * duration["sequence"], duration["step"])
        + iteration_offset
        + duration["offset_gen"]
    )

    params_gen_spk_in = [
        dict(spike_times=timeline_task[input_spike_bools_arr[:, nrn_in_idx]].astype(dtype_in_spks))
        for nrn_in_idx in range(n_in)
    ]

    target_rate_changes = np.zeros((n_out, n_iter_curr * batch_size))
    target_rate_changes[target_cues_arr, np.arange(n_iter_curr * batch_size)] = 1

    params_gen_rate_target = [
        dict(
            amplitude_times=np.arange(0.0, n_iter_curr * batch_size * duration["sequence"], duration["sequence"])
            + iteration_offset
            + duration["total_offset"],
            amplitude_values=target_rate_changes[nrn_out_idx],
        )
        for nrn_out_idx in range(n_out)
    ]

    return params_gen_spk_in, params_gen_rate_target


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

        params_gen_spk_in, params_gen_rate_target = get_params_task_input_output(self.n_iter_sim, n_iter)
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

        if cfg["record_dynamics"]:
            for recorder in dynamics_recorders:
                recorder.set(start=nest.biological_time, stop=nest.biological_time + duration["sequence"])
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

        self.simulate(duration["final_update"], "final_update_0_")
        duration["sim"] += duration["final_update"]

        if comm.rank == 0:
            tools.clear_events("final_update_")

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

# %% ###########################################################################################################
# Save recordings
# ~~~~~~~~~~~~~~~
# We can also retrieve the recorded history of the dynamic variables and weights, as well as detected spikes.

if cfg["record_dynamics"]:
    tools.save_recordings("multimeter_out")
    tools.save_recordings("multimeter_reg")
    tools.save_recordings("multimeter_ad")
    tools.save_recordings("spike_recorder_in")
    tools.save_recordings("spike_recorder_reg")
    tools.save_recordings("spike_recorder_ad")

if cfg["record_weights"]:
    tools.save_recordings("weight_recorder")

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
