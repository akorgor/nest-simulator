# -*- coding: utf-8 -*-
#
# astrocyte_brunel.py
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

"""
Random balanced network with astrocytes
------------------------------------------------------------

This script simulates a network with excitatory and inhibitory neurons and
astrocytes.

"""

###############################################################################
# Import all necessary modules for simulation, analysis and plotting. Scipy
# should be imported before nest.

import time
import multiprocessing as mp

import nest
import nest.raster_plot
import matplotlib.pyplot as plt

###############################################################################
# Simulation parameters

sim_params = {
    "dt": 0.2, # simulation resolution
    "pre_sim_time": 200.0, # time before simulation
    "sim_time": 1000.0, # simulation time
    "N_rec": 100, # number of neurons recorded
    }

###############################################################################
# Network parameters

network_params = {
    "N_ex": 8000, # number of excitatory neurons
    "N_in": 2000, # number of inhibitory neurons
    "N_astro": 10000, # number of astrocytes
    "p": 0.1, # neuron-neuron connection probability.
    "p_syn_astro": 1.0, # synapse-astrocyte pairing probability
    "max_astro_per_target": 1, # Max number of astrocytes per target neuron
    "poisson_rate": 500, # rate of poisson input
    "poisson_prob": 1, # connection probability of poisson input
    }

###############################################################################
# Astrocyte parameters

astro_params = {
    'Ca_tot_astro': 2.0, # Total free astrocytic calcium concentration in uM
    'IP3_0_astro': 0.16, # Baseline value of the astrocytic IP3 concentration in uM
    'K_act_astro': 0.08234, # Astrocytic IP3R dissociation constant of calcium (activation) in uM
    'K_inh_astro': 1.049, # Astrocytic IP3R dissociation constant of calcium (inhibition) in uM
    'K_IP3_1_astro': 0.13, # Astrocytic IP3R dissociation constant of IP3 in uM
    'K_IP3_2_astro': 0.9434, # Astrocytic IP3R dissociation constant of IP3 in uM
    'K_SERCA_astro': 0.1, # Activation constant of astrocytic SERCA pump in uM
    'r_ER_cyt_astro': 0.185, # Ratio between astrocytic ER and cytosol volumes
    'r_IP3_astro': 0.1, # Rate constant of astrocytic IP3 production in uM/ms
    'r_IP3R_astro': 0.001, # Astrocytic IP3R binding constant for calcium inhibition in 1/(uM*ms)
    'r_L_astro': 0.00011, # Rate constant for calcium leak from the astrocytic ER to cytosol in 1/ms
    'tau_IP3_astro': 300.0, # Time constant of astrocytic IP3 degradation
    'v_IP3R_astro': 0.006, # Maximum rate of calcium release via astrocytic IP3R in 1/ms
    'v_SERCA_astro': 0.0009, # Maximum rate of calcium uptake by astrocytic IP3R in uM/ms
    }

###############################################################################
# Neuron parameters

neuron_model = "aeif_cond_alpha_astro"
tau_syn_ex = 2.0
tau_syn_in = 5.0

neuron_params_ex = {
    "V_m": -70.0, # membrane potential in mV
    "C_m": 100.0, # capacitance of membrane in pF
    "t_ref": 2.5, # duration of refractory period in ms
    "V_reset": -61.0,  # reset value for V_m in mV
    "E_L": -70.0, # leak reverse potential in mV
    "g_L": 10.0, # leak conductance in nS
    "a": 0.254, # subthreshold adaptation in nS
    "b": 1.38, # spike-triggered adaptation in pA
    "Delta_T": 2.0, # slope factor in mV
    "tau_w": 115.51, # adaptation time constant in ms
    "V_th": -40.0, # spike initiation threshold in mV
    "V_peak": 0.0, # spike detection threshold in mV
    "E_ex": 0.0, # excitatory reversal potential in mV
    "E_in": -80.0, # inhibitory reversal potential in mV
    "tau_syn_ex": tau_syn_ex, # excitatory synaptic time constant in ms
    "tau_syn_in": tau_syn_in, # inhibitory synaptic time constant in ms
    }

neuron_params_in = {
    "V_m": -70.0, # membrane potential in mV
    "C_m": 100.0, # capacitance of membrane in pF
    "t_ref": 2.5, # duration of refractory period in ms
    "V_reset": -47.21,  # reset value for V_m in mV
    "E_L": -70.0, # leak reverse potential in mV
    "g_L": 10.0, # leak conductance in nS
    "a": 0.254, # subthreshold adaptation in nS
    "b": 1.481, # spike-triggered adaptation in pA
    "Delta_T": 2.0, # slope factor in mV
    "tau_w": 202.386, # adaptation time constant in ms
    "V_th": -40.0, # spike initiation threshold in mV
    "V_peak": 0.0, # spike detection threshold in mV
    "E_ex": 0.0, # excitatory reversal potential in mV
    "E_in": -80.0, # inhibitory reversal potential in mV
    "tau_syn_ex": tau_syn_ex, # excitatory synaptic time constant in ms
    "tau_syn_in": tau_syn_in, # inhibitory synaptic time constant in ms
    }

syn_params = {
    "J_ee": 2.67436, # excitatory-to-excitatory synaptic weight in nS
    "J_ei": 1.05594, # excitatory-to-inhibitory synaptic weight in nS
    "J_ie": -5.96457, # inhibitory-to-excitatory synaptic weight in nS
    "J_ii": -3.58881, # inhibitory-to-inhibitory synaptic weight in nS
    "tau_rec_ee": 2882.9445, # excitatory-to-excitatory depression time constant in ms
    "tau_rec_ei": 5317.747, # excitatory-to-inhibitory depression time constant in ms
    "tau_rec_ie": 226.859, # inhibitory-to-excitatory depression time constant in ms
    "tau_rec_ii": 2542.207,  # inhibitory-to-inhibitory depression time constant in ms
    "tau_fac_ee": 0.0, # excitatory-to-excitatory facilitation time constant in ms
    "tau_fac_ei": 0.0, # excitatory-to-inhibitory facilitation time constant in ms
    "tau_fac_ie": 0.0, # inhibitory-to-excitatory facilitation time constant in ms
    "tau_fac_ii": 0.0,  # inhibitory-to-inhibitory facilitation time constant in ms
    "U_ee": 0.928, # excitatory-to-excitatory release probability parameter
    "U_ei": 0.264, # excitatory-to-inhibitory release probability parameter
    "U_ie": 0.541, # inhibitory-to-excitatory release probability parameter
    "U_ii": 0.189, # inhibitory-to-inhibitory release probability parameter
    }

###############################################################################
# Build functions

def build_create(scale, poisson_time):
    print("Creating nodes")
    nodes_ex = nest.Create(
        neuron_model, network_params["N_ex"]*scale, params=neuron_params_ex)
    nodes_in = nest.Create(
        neuron_model, network_params["N_in"]*scale, params=neuron_params_in)
    nodes_astro = nest.Create(
        "astrocyte", network_params["N_astro"]*scale, params=astro_params)
    noise = nest.Create(
        "poisson_generator",
        params={
            "rate": network_params["poisson_rate"],
            "start": 0.0, "stop": poisson_time
            }
        )
    return nodes_ex, nodes_in, nodes_astro, noise

def build_connect(nodes_ex, nodes_in, nodes_astro, noise):
    print("Connecting poisson")
    # note: result of fixed outdegree is less synchronou than one-to-all
    conn_spec_ne = {
        "rule": "fixed_outdegree",
        "outdegree": int(len(nodes_ex)*network_params["poisson_prob"])
        }
    conn_spec_ni = {
        "rule": "fixed_outdegree",
        "outdegree": int(len(nodes_in)*network_params["poisson_prob"])
        }
    syn_params_ne = {
        "synapse_model": "static_synapse", "weight": syn_params["J_ee"]
        }
    syn_params_ni = {
        "synapse_model": "static_synapse", "weight": syn_params["J_ei"]
        }
    nest.Connect(
        noise, nodes_ex, conn_spec=conn_spec_ne, syn_spec=syn_params_ne
        )
    nest.Connect(
        noise, nodes_in, conn_spec=conn_spec_ni, syn_spec=syn_params_ni
        )

    print("Connecting excitatory")
    conn_params_ex = {
        "rule": "pairwise_bernoulli_astro",
        "astrocyte": nodes_astro,
        "p": network_params["p"],
        "p_syn_astro": network_params["p_syn_astro"],
        "max_astro_per_target": network_params["max_astro_per_target"],
        "astro_pool_per_target_det": False
        }
    syn_params_ee = {
        "synapse_model": "tsodyks_synapse",
        "weight": syn_params["J_ee"],
        "U": syn_params["U_ee"],
        "tau_psc": tau_syn_ex,
        "tau_fac": syn_params["tau_fac_ee"],
        "tau_rec": syn_params["tau_rec_ee"],
        }
    syn_params_ei = {
        "synapse_model": "tsodyks_synapse",
        "weight": syn_params["J_ei"],
        "U": syn_params["U_ei"],
        "tau_psc": tau_syn_ex,
        "tau_fac": syn_params["tau_fac_ei"],
        "tau_rec": syn_params["tau_rec_ei"],
        }
    nest.Connect(nodes_ex, nodes_ex, conn_params_ex, syn_params_ee)
    nest.Connect(nodes_ex, nodes_in, conn_params_ex, syn_params_ei)

    print("Connecting inhibitory")
    conn_params_in = {"rule": "pairwise_bernoulli", "p": network_params["p"]}
    syn_params_ie = {
        "synapse_model": "tsodyks_synapse",
        "weight": syn_params["J_ie"],
        "U": syn_params["U_ie"],
        "tau_psc": tau_syn_in,
        "tau_fac": syn_params["tau_fac_ie"],
        "tau_rec": syn_params["tau_rec_ie"],
        }
    syn_params_ii = {
        "synapse_model": "tsodyks_synapse",
        "weight": syn_params["J_ii"],
        "U": syn_params["U_ii"],
        "tau_psc": tau_syn_in,
        "tau_fac": syn_params["tau_fac_ii"],
        "tau_rec": syn_params["tau_rec_ii"],
        }
    nest.Connect(nodes_in, nodes_ex, conn_params_in, syn_params_ie)
    nest.Connect(nodes_in, nodes_in, conn_params_in, syn_params_ii)

###############################################################################
# Main function for running simulation.

def run_simulation():
    # reset kernel
    nest.ResetKernel()
    nest.SetKernelStatus({'total_num_virtual_procs': int(mp.cpu_count()/2)})

    # time before building
    startbuild = time.time()

    # NEST configuration
    nest.resolution = sim_params["dt"]
    nest.print_time = True
    nest.overwrite_files = True

    # create and connect nodes
    total_time = sim_params["pre_sim_time"] + sim_params["sim_time"]
    e, i, a, n = build_create(scale=1, poisson_time=total_time)
    build_connect(e, i, a, n)

    # create spike recorder
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")
    espikes.set(label="astro-ex", record_to="ascii")
    ispikes.set(label="astro-in", record_to="ascii")
    nest.Connect(e[:sim_params["N_rec"]], espikes)
    nest.Connect(i[:sim_params["N_rec"]], ispikes)

    # time after building
    endbuild = time.time()

    # simulation
    print("Simulating")
    nest.Simulate(sim_params["pre_sim_time"])
    nest.Simulate(sim_params["sim_time"])

    # time after simulation
    endsimulate = time.time()

    # read out spikes and calculate firing rates
    events_ex = espikes.n_events
    events_in = ispikes.n_events
    rate_ex = events_ex / sim_params["sim_time"] * 1000.0 / sim_params["N_rec"]
    rate_in = events_in / sim_params["sim_time"] * 1000.0 / sim_params["N_rec"]

    # calculate building and running time
    build_time = endbuild - startbuild
    run_time = endsimulate - endbuild

    # print firing rates and building and running time
    print("Brunel network with astrocytes")
    print(f"Excitatory rate   : {rate_ex:.2f} Hz")
    print(f"Inhibitory rate   : {rate_in:.2f} Hz")
    print(f"Building time     : {build_time:.2f} s")
    print(f"Simulation time   : {run_time:.2f} s")

    # plot a raster of the excitatory neurons and a histogram
    nest.raster_plot.from_device(espikes, hist=True)
    plt.savefig("astrocyte_brunel.png")

if __name__ == "__main__":
    run_simulation()
