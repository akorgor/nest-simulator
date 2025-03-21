/*
 *  nest_names.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef NEST_NAMES_H
#define NEST_NAMES_H

// Generated includes:
#include "config.h"

// Includes from sli:
#include "name.h"

namespace nest
{

/**
 * This namespace contains global Name objects.
 *
 * These can be used in Node::get_status and Node::set_status to make data exchange more
 * efficient and consistent. Creating a Name from a std::string is in
 * O(log n), for n the number of Names already created. Using
 * predefined names makes data exchange much more efficient as it
 * uses integer comparisons instead of string comparisons internally.
 *
 * The Name declarations below and the definitions in nest_names.cpp
 * are sorted alphabetically with lower case letters preceding upper
 * case letters. The ordering of the names has to be the same in both
 * this file and the .cpp file.
 *
 * See testsuite/unittests/test_unused_names.py for a test that checks
 * if a) Name declarations and definitions are consistent
 *    b) all Name objects defined are also actually used.
 */
namespace names
{
extern const Name A_LTD;
extern const Name A_LTD_const;
extern const Name A_LTP;
extern const Name A_minus;
extern const Name A_plus;
extern const Name Act_m;
extern const Name Act_n;
extern const Name AMPA;
extern const Name Aminus;
extern const Name Aminus_triplet;
extern const Name Aplus;
extern const Name Aplus_triplet;
extern const Name ASCurrents;
extern const Name ASCurrents_sum;
extern const Name a;
extern const Name a_acausal;
extern const Name a_causal;
extern const Name a_thresh_th;
extern const Name a_thresh_tl;
extern const Name acceptable_latency;
extern const Name activity;
extern const Name adapt_beta;
extern const Name adapt_tau;
extern const Name adaptation;
extern const Name adapting_threshold;
extern const Name adaptive_target_buffers;
extern const Name add_compartments;
extern const Name add_receptors;
extern const Name after_spike_currents;
extern const Name ahp_bug;
extern const Name allow_autapses;
extern const Name allow_multapses;
extern const Name allow_offgrid_times;
extern const Name allow_oversized_mask;
extern const Name alpha;
extern const Name alpha_1;
extern const Name alpha_2;
extern const Name amp_slow;
extern const Name amplitude;
extern const Name amplitude_times;
extern const Name amplitude_values;
extern const Name anchor;
extern const Name archiver_length;
extern const Name asc_amps;
extern const Name asc_decay;
extern const Name asc_init;
extern const Name asc_r;
extern const Name available;
extern const Name average_gradient;
extern const Name azimuth_angle;

extern const Name b;
extern const Name batch_size;
extern const Name beta;
extern const Name beta_1;
extern const Name beta_2;

extern const Name beta_Ca;
extern const Name biological_time;
extern const Name box;
extern const Name buffer_size;
extern const Name buffer_size_spike_data;
extern const Name buffer_size_target_data;

extern const Name C_m;
extern const Name Ca;
extern const Name Ca_astro;
extern const Name Ca_tot;
extern const Name c;
extern const Name c_1;
extern const Name c_2;
extern const Name c_3;
extern const Name c_reg;
extern const Name capacity;
extern const Name center;
extern const Name circular;
extern const Name clear;
extern const Name comp_idx;
extern const Name comparator;
extern const Name compartments;
extern const Name conc_Mg2;
extern const Name configbit_0;
extern const Name configbit_1;
extern const Name connection_count;
extern const Name connection_rules;
extern const Name connection_type;
extern const Name consistent_integration;
extern const Name continuous;
extern const Name count_covariance;
extern const Name count_histogram;
extern const Name covariance;

extern const Name Delta_T;
extern const Name Delta_V;
extern const Name d;
extern const Name data;
extern const Name data_path;
extern const Name data_prefix;
extern const Name dead_time;
extern const Name dead_time_random;
extern const Name dead_time_shape;
extern const Name delay;
extern const Name delay_u_bars;
extern const Name deliver_interval;
extern const Name delta;
extern const Name delta_IP3;
extern const Name delta_P;
extern const Name delta_tau;
extern const Name dendritic_curr;
extern const Name dendritic_exc;
extern const Name dendritic_inh;
extern const Name dg;
extern const Name dg_ex;
extern const Name dg_in;
extern const Name dI_syn_ex;
extern const Name dI_syn_in;
extern const Name dict_miss_is_error;
extern const Name diffusion_factor;
extern const Name dimension;
extern const Name distal_curr;
extern const Name distal_exc;
extern const Name distal_inh;
extern const Name drift_factor;
extern const Name driver_readout_time;
extern const Name dt;
extern const Name dU;

extern const Name E_ahp;
extern const Name E_ex;
extern const Name E_in;
extern const Name E_K;
extern const Name E_L;
extern const Name E_Na;
extern const Name E_rev;
extern const Name E_rev_AMPA;
extern const Name E_rev_GABA_A;
extern const Name E_rev_GABA_B;
extern const Name E_rev_h;
extern const Name E_rev_KNa;
extern const Name E_rev_NaP;
extern const Name E_rev_NMDA;
extern const Name E_rev_T;
extern const Name E_rr;
extern const Name E_sfa;
extern const Name e_L;
extern const Name edge_wrap;
extern const Name element_type;
extern const Name elements;
extern const Name elementsize;
extern const Name ellipsoidal;
extern const Name elliptical;
extern const Name eprop_history_duration;
extern const Name eprop_isi_trace_cutoff;
extern const Name eprop_learning_window;
extern const Name eprop_reset_neurons_on_update;
extern const Name eprop_update_interval;
extern const Name eps;
extern const Name epsilon;

extern const Name equilibrate;
extern const Name error_signal;
extern const Name eta;
extern const Name events;
extern const Name extent;

extern const Name f_target;
extern const Name file_extension;
extern const Name filename;
extern const Name filenames;
extern const Name frequency;
extern const Name frozen;

extern const Name GABA;
extern const Name GABA_A;
extern const Name GABA_B;
extern const Name g;
extern const Name g_AMPA;
extern const Name g_ahp;
extern const Name g_C;
extern const Name g_ex;
extern const Name g_GABA_A;
extern const Name g_GABA_B;
extern const Name g_in;
extern const Name g_K;
extern const Name g_KL;
extern const Name g_Kv1;
extern const Name g_Kv3;
extern const Name g_L;
extern const Name g_m;
extern const Name g_Na;
extern const Name g_NaL;
extern const Name g_NMDA;
extern const Name g_pd;
extern const Name g_peak_AMPA;
extern const Name g_peak_GABA_A;
extern const Name g_peak_GABA_B;
extern const Name g_peak_h;
extern const Name g_peak_KNa;
extern const Name g_peak_NaP;
extern const Name g_peak_NMDA;
extern const Name g_peak_T;
extern const Name g_ps;
extern const Name g_rr;
extern const Name g_sfa;
extern const Name g_sp;
extern const Name gamma;
extern const Name gamma_shape;
extern const Name gaussian;
extern const Name global_id;
extern const Name grid;
extern const Name grid3d;
extern const Name growth_curve;
extern const Name growth_curves;
extern const Name growth_factor_buffer_spike_data;
extern const Name growth_factor_buffer_target_data;
extern const Name growth_rate;
extern const Name gsl_error_tol;

extern const Name h;
extern const Name h_IP3R;
extern const Name has_connections;
extern const Name has_delay;
extern const Name histogram;
extern const Name histogram_correction;

extern const Name I;
extern const Name I_ahp;
extern const Name I_e;
extern const Name I_h;
extern const Name I_AMPA;
extern const Name I_GABA;
extern const Name I_KNa;
extern const Name I_NMDA;
extern const Name I_NaP;
extern const Name I_SIC;
extern const Name I_sp;
extern const Name I_stc;
extern const Name I_syn;
extern const Name I_syn_ex;
extern const Name I_syn_in;
extern const Name I_T;
extern const Name Inact_h;
extern const Name Inact_p;
extern const Name IP3;
extern const Name IP3_0;
extern const Name indegree;
extern const Name index_map;
extern const Name individual_spike_trains;
extern const Name init_flag;
extern const Name inner_radius;
extern const Name instant_unblock_NMDA;
extern const Name instantiations;
extern const Name interval;
extern const Name is_refractory;

extern const Name kappa;
extern const Name kappa_reg;
extern const Name Kd_act;
extern const Name Kd_IP3_1;
extern const Name Kd_IP3_2;
extern const Name Kd_inh;
extern const Name Km_SERCA;
extern const Name Kplus;
extern const Name Kplus_triplet;
extern const Name k_IP3R;
extern const Name keep_source_table;
extern const Name kernel;

extern const Name label;
extern const Name lambda;
extern const Name lambda_0;
extern const Name learning_signal;
extern const Name len_kernel;
extern const Name linear;
extern const Name linear_summation;
extern const Name local;
extern const Name local_num_threads;
extern const Name local_spike_counter;
extern const Name lookuptable_0;
extern const Name lookuptable_1;
extern const Name lookuptable_2;
extern const Name loss;
extern const Name lower_left;

extern const Name m;
extern const Name major_axis;
extern const Name make_symmetric;
extern const Name mask;
extern const Name max;
extern const Name max_buffer_size_target_data;
extern const Name max_delay;
extern const Name max_num_syn_models;
extern const Name max_update_time;
extern const Name mean;
extern const Name memory;
extern const Name message_times;
extern const Name messages;
extern const Name min;
extern const Name min_delay;
extern const Name min_update_time;
extern const Name minor_axis;
extern const Name model;
extern const Name model_id;
extern const Name modules;
extern const Name mpi_address;
extern const Name ms_per_tic;
extern const Name mu;
extern const Name mu_minus;
extern const Name mu_plus;
extern const Name mult_coupling;
extern const Name music_channel;

extern const Name N;
extern const Name NMDA;
extern const Name N_channels;
extern const Name N_NaP;
extern const Name N_T;
extern const Name n;
extern const Name n_events;
extern const Name n_messages;
extern const Name n_proc;
extern const Name n_receptors;
extern const Name n_synapses;
extern const Name network_size;
extern const Name neuron;
extern const Name next_readout_time;
extern const Name no_synapses;
extern const Name node_models;
extern const Name node_uses_wfr;
extern const Name noise;
extern const Name noisy_rate;
extern const Name num_connections;
extern const Name num_processes;
extern const Name number_of_connections;

extern const Name off_grid_spiking;
extern const Name offset;
extern const Name offsets;
extern const Name omega;
extern const Name optimizer;
extern const Name optimize_each_step;
extern const Name order;
extern const Name origin;
extern const Name other;
extern const Name outdegree;
extern const Name outer_radius;
extern const Name overwrite_files;

extern const Name P;
extern const Name p;
extern const Name p_copy;
extern const Name p_transmit;
extern const Name pairwise_bernoulli_on_source;
extern const Name pairwise_bernoulli_on_target;
extern const Name pairwise_avg_num_conns;
extern const Name params;
extern const Name parent_idx;
extern const Name phase;
extern const Name phi_max;
extern const Name pairwise_poisson;
extern const Name polar_angle;
extern const Name polar_axis;
extern const Name pool_size;
extern const Name pool_type;
extern const Name port;
extern const Name port_name;
extern const Name port_width;
extern const Name ports;
extern const Name positions;
extern const Name post_synaptic_element;
extern const Name post_trace;
extern const Name pre_synaptic_element;
extern const Name precise_times;
extern const Name precision;
extern const Name prepared;
extern const Name primary;
extern const Name print_time;
extern const Name proximal_curr;
extern const Name proximal_exc;
extern const Name proximal_inh;
extern const Name psi;
extern const Name published;
extern const Name pulse_times;

extern const Name q_rr;
extern const Name q_sfa;
extern const Name q_stc;

extern const Name radius;
extern const Name rate;
extern const Name rate_IP3R;
extern const Name rate_L;
extern const Name rate_SERCA;
extern const Name rate_slope;
extern const Name rate_times;
extern const Name rate_values;
extern const Name ratio_ER_cyt;
extern const Name readout_cycle_duration;
extern const Name readout_signal;
extern const Name readout_signal_unnorm;
extern const Name receptor_idx;
extern const Name receptor_type;
extern const Name receptor_types;
extern const Name receptors;
extern const Name record_from;
extern const Name record_to;
extern const Name recordables;
extern const Name recorder;
extern const Name recording_backends;
extern const Name rectangular;
extern const Name rectify_output;
extern const Name rectify_rate;
extern const Name recv_buffer_size_secondary_events;
extern const Name refractory_input;
extern const Name registered;
extern const Name regular_spike_arrival;
extern const Name relative_amplitude;
extern const Name requires_symmetric;
extern const Name reset_pattern;
extern const Name resolution;
extern const Name rho;
extern const Name rng_seed;
extern const Name rng_type;
extern const Name rng_types;
extern const Name rport;
extern const Name rule;

extern const Name S;
extern const Name S_act_NMDA;
extern const Name s_GABA;
extern const Name s_AMPA;
extern const Name s_NMDA;
extern const Name SIC_scale;
extern const Name SIC_th;
extern const Name sdev;
extern const Name send_buffer_size_secondary_events;
extern const Name senders;
extern const Name shape;
extern const Name shift_now_spikes;
extern const Name shrink_factor_buffer_spike_data;
extern const Name sigma;
extern const Name sigmoid;
extern const Name sion_chunksize;
extern const Name sion_collective;
extern const Name sion_n_files;
extern const Name size_of;
extern const Name soma_curr;
extern const Name soma_exc;
extern const Name soma_inh;
extern const Name source;
extern const Name spherical;
extern const Name spike_buffer_grow_extra;
extern const Name spike_buffer_resize_log;
extern const Name spike_buffer_shrink_limit;
extern const Name spike_buffer_shrink_spare;
extern const Name spike_dependent_threshold;
extern const Name spike_multiplicities;
extern const Name spike_times;
extern const Name spike_weights;
extern const Name start;
extern const Name state;
extern const Name std;
extern const Name std_mod;
extern const Name stimulation_backends;
extern const Name stimulator;
extern const Name stimulus_source;
extern const Name stop;
extern const Name structural_plasticity_synapses;
extern const Name structural_plasticity_update_interval;
extern const Name surrogate_gradient;
extern const Name surrogate_gradient_function;
extern const Name synapse_id;
extern const Name synapse_label;
extern const Name synapse_model;
extern const Name synapse_modelid;
extern const Name synapse_models;
extern const Name synapse_parameters;
extern const Name synapses_per_driver;
extern const Name synaptic_elements;
extern const Name synaptic_elements_param;
extern const Name synaptic_endpoint;

extern const Name T_max;
extern const Name T_min;
extern const Name Tstart;
extern const Name Tstop;
extern const Name t_clamp;
extern const Name t_ref;
extern const Name t_ref_abs;
extern const Name t_ref_remaining;
extern const Name t_ref_tot;
extern const Name t_spike;
extern const Name target;
extern const Name target_signal;
extern const Name target_thread;
extern const Name targets;
extern const Name tau;
extern const Name tau_1;
extern const Name tau_2;
extern const Name tau_AMPA;
extern const Name tau_GABA;
extern const Name tau_ahp;
extern const Name tau_Ca;
extern const Name tau_c;
extern const Name tau_D_KNa;
extern const Name tau_Delta;
extern const Name tau_decay;
extern const Name tau_decay_AMPA;
extern const Name tau_decay_ex;
extern const Name tau_decay_GABA_A;
extern const Name tau_decay_GABA_B;
extern const Name tau_decay_in;
extern const Name tau_decay_NMDA;
extern const Name tau_epsp;
extern const Name tau_fac;
extern const Name tau_IP3;
extern const Name tau_Mg_fast_NMDA;
extern const Name tau_Mg_slow_NMDA;
extern const Name tau_m;
extern const Name tau_max;
extern const Name tau_minus;
extern const Name tau_minus_stdp;
extern const Name tau_minus_triplet;
extern const Name tau_m_readout;
extern const Name tau_n;
extern const Name tau_P;
extern const Name tau_plus;
extern const Name tau_plus_triplet;
extern const Name tau_psc;
extern const Name tau_rec;
extern const Name tau_reset;
extern const Name tau_rise;
extern const Name tau_rise_AMPA;
extern const Name tau_rise_ex;
extern const Name tau_rise_GABA_A;
extern const Name tau_rise_GABA_B;
extern const Name tau_rise_in;
extern const Name tau_rise_NMDA;
extern const Name tau_rr;
extern const Name tau_sfa;
extern const Name tau_spike;
extern const Name tau_stc;
extern const Name tau_syn;
extern const Name tau_syn_ex;
extern const Name tau_syn_fast;
extern const Name tau_syn_in;
extern const Name tau_syn_slow;
extern const Name tau_theta;
extern const Name tau_u_bar_bar;
extern const Name tau_u_bar_minus;
extern const Name tau_u_bar_plus;
extern const Name tau_V_th;
extern const Name tau_v;
extern const Name tau_vacant;
extern const Name tau_w;
extern const Name tau_x;
extern const Name tau_z;
extern const Name th_spike_add;
extern const Name th_spike_decay;
extern const Name th_voltage_decay;
extern const Name th_voltage_index;
extern const Name theta;
extern const Name theta_eq;
extern const Name theta_ex;
extern const Name theta_in;
extern const Name theta_minus;
extern const Name theta_plus;
extern const Name third_in;
extern const Name third_out;
extern const Name thread;
extern const Name thread_local_id;
extern const Name threshold;
extern const Name threshold_spike;
extern const Name threshold_voltage;
extern const Name tics_per_ms;
extern const Name tics_per_step;
extern const Name time_collocate_spike_data;
extern const Name time_collocate_spike_data_cpu;
extern const Name time_communicate_prepare;
extern const Name time_communicate_prepare_cpu;
extern const Name time_communicate_spike_data;
extern const Name time_communicate_spike_data_cpu;
extern const Name time_communicate_target_data;
extern const Name time_communicate_target_data_cpu;
extern const Name time_construction_connect;
extern const Name time_construction_connect_cpu;
extern const Name time_construction_create;
extern const Name time_construction_create_cpu;
extern const Name time_deliver_secondary_data;
extern const Name time_deliver_secondary_data_cpu;
extern const Name time_deliver_spike_data;
extern const Name time_deliver_spike_data_cpu;
extern const Name time_gather_secondary_data;
extern const Name time_gather_secondary_data_cpu;
extern const Name time_gather_spike_data;
extern const Name time_gather_spike_data_cpu;
extern const Name time_gather_target_data;
extern const Name time_gather_target_data_cpu;
extern const Name time_omp_synchronization_construction;
extern const Name time_omp_synchronization_construction_cpu;
extern const Name time_omp_synchronization_simulation;
extern const Name time_omp_synchronization_simulation_cpu;
extern const Name time_mpi_synchronization;
extern const Name time_mpi_synchronization_cpu;
extern const Name time_in_steps;
extern const Name time_simulate;
extern const Name time_simulate_cpu;
extern const Name time_update;
extern const Name time_update_cpu;
extern const Name times;
extern const Name to_do;
extern const Name total_num_virtual_procs;
extern const Name type;
extern const Name type_id;

extern const Name U;
extern const Name U_m;
extern const Name u;
extern const Name u_bar_bar;
extern const Name u_bar_minus;
extern const Name u_bar_plus;
extern const Name u_ref_squared;
extern const Name update_time_limit;
extern const Name upper_right;
extern const Name use_compressed_spikes;
extern const Name use_wfr;

extern const Name v;
extern const Name V_act_NMDA;
extern const Name V_clamp;
extern const Name v_comp;
extern const Name V_epsp;
extern const Name V_m;
extern const Name V_min;
extern const Name V_noise;
extern const Name V_peak;
extern const Name V_reset;
extern const Name V_T;
extern const Name V_T_star;
extern const Name V_th;
extern const Name V_th_adapt;
extern const Name V_th_alpha_1;
extern const Name V_th_alpha_2;
extern const Name V_th_max;
extern const Name V_th_rest;
extern const Name V_th_v;
extern const Name voltage_clamp;
extern const Name voltage_reset_add;
extern const Name voltage_reset_fraction;
extern const Name volume_transmitter;
extern const Name vp;

extern const Name Wmax;
extern const Name Wmin;
extern const Name w;
extern const Name weight;
extern const Name weight_per_lut_entry;
extern const Name weight_recorder;
extern const Name weights;
extern const Name wfr_comm_interval;
extern const Name wfr_interpolation_order;
extern const Name wfr_max_iterations;
extern const Name wfr_tol;
extern const Name with_reset;

extern const Name x;
extern const Name x_bar;

extern const Name y;
extern const Name y_0;
extern const Name y_1;

extern const Name z;
extern const Name z_connected;
} // namespace names

} // namespace nest

#endif /* #ifndef NEST_NAMES_H */
