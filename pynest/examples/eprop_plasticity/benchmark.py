from pathlib import Path
import nest
import json
import numpy as np

base_path = Path(__file__).parent

with open(base_path / "config.json", "r") as f:
    cfg = json.load(f)

local_num_threads = cfg["cpus_per_task"]
total_num_virtual_procs = cfg["nodes"] * cfg["ntasks_per_node"] * local_num_threads

n_rec = int(cfg["n_rec_base"]*cfg["scale"])
n_in = int(cfg["n_in_base"]*cfg["scale"])
n_out = int(cfg["n_out_base"]*cfg["scale"])

dt = 1.0
duration = cfg["steps"] * dt

offset = 2.0

if cfg["plasticity"] == "eprop_bio":
    model_syn_rec = "eprop_synapse"
    model_nrn_rec = "eprop_iaf"
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
    model_nrn_rec = "ignore_and_fire"
    model_nrn_out = "iaf_psc_delta"
    model_conn_fb = None
    model_gen_rate = None
elif cfg["plasticity"] == "stdp":
    model_syn_rec = "stdp_synapse"
    model_nrn_rec = "ignore_and_fire"
    model_nrn_out = "iaf_psc_delta"
    model_conn_fb = None
    model_gen_rate = None

rng = np.random.default_rng(cfg["seed"])

nest.ResetKernel()
nest.set(resolution=dt, rng_seed=cfg["seed"], local_num_threads=local_num_threads, total_num_virtual_procs=total_num_virtual_procs, eprop_update_interval=cfg["eprop_update_interval"])

nrns_rec = nest.Create(model_nrn_rec, n_rec)
nrns_out = nest.Create(model_nrn_out, n_out)

if n_in > 0:
    if cfg["input_generator"] == "poisson_generator":
        pg = nest.Create("poisson_generator", n_in, dict(rate=cfg["rate_in"]))
    elif cfg["input_generator"] == "spike_generator":
            counts = rng.poisson(cfg["rate_in"] * duration / 1000.0, size=n_in)
            total_spikes = counts.sum()
            times_all = rng.integers(offset, duration+offset, size=total_spikes)

            offsets = np.cumsum(np.r_[0, counts])
            for start, end in zip(offsets[:-1], offsets[1:]):
                times_all[start:end].sort()

            times_split = np.split(times_all, offsets[1:-1])
            sg_params = [{"spike_times": arr.astype(float)} for arr in times_split]

            nrns_in = nest.Create("spike_generator", n_in, sg_params)

    nest.Connect(nrns_in, nrns_rec, dict(rule="fixed_indegree", indegree=cfg["indegree_in"]), dict(synapse_model="static_synapse"))

if "eprop" in cfg["plasticity"]:
    nest.SetDefaults(model_syn_rec)
    nest.SetStatus(nrns_rec, dict(ignore_and_fire=True, phase=nest.random.uniform(0.0, 1.0), rate=cfg["rate_rec"]))

if model_nrn_out == "iaf_psc_delta":
    nest.SetStatus(nrns_out, dict(V_th=1e100))

nest.Connect(nrns_rec, nrns_rec, dict(rule="fixed_indegree", indegree=cfg["indegree_rec"]), dict(synapse_model=model_syn_rec))
nest.Connect(nrns_rec, nrns_out, dict(rule="fixed_indegree", indegree=cfg["indegree_out"]), dict(synapse_model="static_synapse"))

if model_conn_fb is not None:
    nest.Connect(nrns_out, nrns_rec, dict(rule="fixed_outdegree", outdegree=cfg["outdegree_fb"]), dict(synapse_model=model_conn_fb))

if model_gen_rate is not None:
    gen_rate_target = nest.Create(model_gen_rate, n_out)
    nest.Connect(gen_rate_target, nrns_out, dict(rule="one_to_one"), dict(synapse_model="rate_connection_delayed", receptor_type=2))

nest.Simulate(duration + offset)

if nest.Rank() == 0:
    import shutil
    import json

    kernel_status = nest.GetKernelStatus()

    results = dict(
        n_spikes=kernel_status["local_spike_counter"],
        time_sim=kernel_status["time_simulate"],
        time_bio=kernel_status["biological_time"],
        cpus_per_task=cfg["cpus_per_task"],
        nodes=cfg["nodes"],
        ntasks_per_node=cfg["ntasks_per_node"],
        plasticity=cfg["plasticity"],
    )

    for key in kernel_status.keys():
        if key.startswith("time"):
            results[key] = kernel_status[key]

    with open(Path(cfg["recordings_dir"]) / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    pycache_path = base_path / "__pycache__"
    if pycache_path.exists() and pycache_path.is_dir():
        shutil.rmtree(pycache_path)