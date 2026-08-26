import json
import pathlib

import nest
from mpi4py import MPI
import numpy as np

memory_log = []

def log_memory(label):
    memory_log.append((nest.Rank(), label, nest.memory_size))

file_path = pathlib.Path(__file__)

with open(file_path.parent / "config.json", "r") as f:
    cfg = json.load(f)

path_recordings_dir = file_path.parent / cfg["relative_path_recordings_dir"]

if nest.Rank() == 0:
    if path_recordings_dir.exists() and any(path_recordings_dir.iterdir()):
        if cfg["delete_existing_recordings"]:
            for f in path_recordings_dir.iterdir():
                if f.is_file():
                    f.unlink()
        else:
            print(
                "\nWARNING: The recordings directory is not empty. This may cause the run to fail if overwriting is disabled or lead to incorrect results.\n"
            )
    else:
        path_recordings_dir.mkdir(parents=True, exist_ok=True)

local_num_threads = cfg["job_cpus_per_task"]
total_num_virtual_procs = cfg["job_nodes"] * cfg["job_ntasks_per_node"] * local_num_threads

n_rec = int(cfg["n_rec_base"] * cfg["scale"])
n_in = int(cfg["n_in_base"] * cfg["scale"])
n_out = int(cfg["n_out_base"] * cfg["scale"])

if cfg["plasticity"] == "eprop_bio":
    model_syn_rec = "eprop_synapse"
    model_nrn_rec = "eprop_iaf_psc_delta"
    model_nrn_out = "eprop_readout"
    model_conn_fb = "eprop_learning_signal_connection"
    model_gen_rate = "step_rate_generator"
elif cfg["plasticity"] == "eprop_original":
    model_syn_rec = "eprop_synapse_bsshslm_2020"
    model_nrn_rec = "eprop_iaf_bsshslm_2020"
    model_nrn_out = "eprop_readout_bsshslm_2020"
    model_conn_fb = "eprop_learning_signal_connection_bsshslm_2020"
    model_gen_rate = "step_rate_generator"
elif cfg["plasticity"] == "static":
    model_syn_rec = "static_synapse"
    model_nrn_rec = "iaf_psc_delta"
    model_nrn_out = "iaf_psc_delta"
    model_conn_fb = None
    model_gen_rate = None
elif cfg["plasticity"] == "stdp":
    model_syn_rec = "stdp_synapse"
    model_nrn_rec = "iaf_psc_delta"
    model_nrn_out = "iaf_psc_delta"
    model_conn_fb = None
    model_gen_rate = None
elif cfg["plasticity"] == "regular":
    model_syn_rec = "static_synapse"
    model_nrn_rec = "iaf_psc_delta"
    model_nrn_out = "iaf_psc_delta"
    model_conn_fb = None
    model_gen_rate = None
else:
    raise ValueError(f"Unknown plasticity type: {cfg['plasticity']}")

rng = np.random.default_rng(cfg["seed"])

nest.ResetKernel()
log_memory("after_reset")
nest.set(
    resolution=1.0,
    rng_seed=cfg["seed"],
    local_num_threads=local_num_threads,
    total_num_virtual_procs=total_num_virtual_procs,
    eprop_update_interval=cfg["eprop_update_interval"],
)

duration = cfg["steps"] * nest.resolution
offset = 2 * nest.resolution

# Create nodes =========================================================================================================
log_memory("after_set")

nrns_rec = nest.Create(model_nrn_rec, n_rec)
nrns_out = nest.Create(model_nrn_out, n_out)

if n_in > 0:
    if cfg["input_generator"] == "poisson_generator":
        spk_gen = nest.Create("poisson_generator", 1, dict(rate=cfg["rate_in"], start=offset, stop=duration + offset))
        nrns_in = nest.Create("parrot_neuron", n_in)
        nest.Connect(spk_gen, nrns_in, dict(rule="all_to_all"), dict(synapse_model="static_synapse"))
    elif cfg["input_generator"] in ["spike_generator", "spike_train_injector"]:
        nrns_in = nest.Create(cfg["input_generator"], n_in)
        local_nrns_in = nest.GetLocalNodeCollection(nrns_in)

        for sg in local_nrns_in:
            sg_rng = np.random.default_rng(cfg["seed"] + sg.global_id)
            count = sg_rng.poisson(cfg["rate_in"] * duration / 1000.0)
            times = sg_rng.integers(int(offset), int(duration + offset), size=count).astype(float)
            times.sort()
            sg.set(spike_times=times)

    else:
        raise ValueError(f"Unknown input_generator type: {cfg['input_generator']}")

if model_gen_rate is not None:
    gen_rate_target = nest.Create(model_gen_rate, n_out)

# Configure nodes ======================================================================================================

if cfg["plasticity"] != "regular":
    ignore_and_spike_interval = 1000.0 / cfg["rate_rec"]  # ms

    # Avoid spikes during the first few time steps, since they would require
    # retrieving parts of the e-prop history that have not yet been populated.
    params = dict(
        ignore_and_spike=True,
        ignore_and_spike_offset=nest.random.uniform(5, ignore_and_spike_interval),
        ignore_and_spike_interval=ignore_and_spike_interval
    )

    if cfg["plasticity"].startswith("eprop"):
        params["flush_event_send_interval"] = cfg["eprop_update_interval"]
    
    if cfg["plasticity"] == "eprop_bio":
        params["eprop_isi_trace_cutoff"] = 100.0
    nrns_rec.set(params)

if model_nrn_out == "iaf_psc_delta":
    nrns_out.set(dict(V_th=1e100))

# Connect nodes ========================================================================================================
log_memory("after_configure")

if n_in > 0:
    nest.Connect(nrns_in, nrns_rec, dict(rule="fixed_indegree", indegree=cfg["indegree_in"]), dict(synapse_model="static_synapse"))

nest.Connect(nrns_rec, nrns_rec, dict(rule="fixed_indegree", indegree=cfg["indegree_rec"]), dict(synapse_model=model_syn_rec))
nest.Connect(nrns_rec, nrns_out, dict(rule="fixed_indegree", indegree=cfg["indegree_out"]), dict(synapse_model="static_synapse"))

if model_conn_fb is not None:
    nest.Connect(nrns_out, nrns_rec, dict(rule="fixed_outdegree", outdegree=cfg["outdegree_fb"]), dict(synapse_model=model_conn_fb))

if model_gen_rate is not None:
    nest.Connect(gen_rate_target, nrns_out, dict(rule="one_to_one"), dict(synapse_model="rate_connection_delayed", receptor_type=2))

# Simulate =============================================================================================================
log_memory("after_connect")
nest.Prepare()
log_memory("after_prepare")
nest.Run(duration + offset)
log_memory("after_run")
nest.Cleanup()
log_memory("after_cleanup")

# Evaluate =============================================================================================================
all_memory_logs = MPI.COMM_WORLD.gather(memory_log, root=0)

if nest.Rank() == 0:
    with open(path_recordings_dir / "memory.csv", "w") as f:
        f.write("rank,stage,memory_kb\n")
        for rank_log in all_memory_logs:
            for rank, stage, memory_kb in rank_log:
                f.write(f"{rank},{stage},{memory_kb}\n")

    kernel_status = nest.GetKernelStatus()

    results = dict(
        n_spikes=kernel_status["local_spike_counter"],
        time_sim=kernel_status["time_simulate"],
        time_bio=kernel_status["biological_time"],
        job_cpus_per_task=cfg["job_cpus_per_task"],
        job_nodes=cfg["job_nodes"],
        job_ntasks_per_node=cfg["job_ntasks_per_node"],
        plasticity=cfg["plasticity"],
    )

    for key in kernel_status.keys():
        if key.startswith("time"):
            results[key] = kernel_status[key]

    for k, v in results.items():
        if isinstance(v, np.ndarray):
            results[k] = v.tolist()

    with open(path_recordings_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
