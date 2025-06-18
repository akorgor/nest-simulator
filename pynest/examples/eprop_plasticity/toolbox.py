import nest
import numpy as np
import pandas as pd
import json
from pathlib import Path


class Tools:
    def __init__(self, config, file_path):
        self.config = config
        self.load_config()
        self.recordings_dir = Path(self.config["recordings_dir"])
        self.save_config()
        self.file_name = Path(file_path).name
        self.remove_recordings()
        self.timing_dict = dict(
            biological_time=0.0,
            time_communicate_prepare=0.0,
            time_construction_connect=0.0,
            time_construction_create=0.0,
            time_simulate=0.0,
        )
        self.loss = []

    def load_config(self):
        with open("config.json") as f:
            config_updated = json.load(f)
            for k, v in config_updated.items():
                self.config[k] = v

    def save_config(self):
        with open(self.recordings_dir / "config.json", "w") as file:
            json.dump(self.config, file)

    def remove_recordings(self):
        for file_path in self.recordings_dir.iterdir():
            if file_path.is_file() and (file_path.suffix in [".csv", ".dat"]):
                file_path.unlink()

    def constrain_weights(self, nrns_in, nrns_rec, nrns_out, params_syn_base, params_common_syn_eprop):
        weight_dicts = [
            dict(
                nrns_pre=nrns_in,
                nrns_post=nrns_rec,
                constrain_sign=self.config["constrain_weights_sign_in"],
                constrain_dale=self.config["constrain_weights_dale_in"],
            ),
            dict(
                nrns_pre=nrns_rec,
                nrns_post=nrns_rec,
                constrain_sign=self.config["constrain_weights_sign_rec"],
                constrain_dale=self.config["constrain_weights_dale_rec"],
            ),
            dict(
                nrns_pre=nrns_rec,
                nrns_post=nrns_out,
                constrain_sign=self.config["constrain_weights_sign_out"],
                constrain_dale=self.config["constrain_weights_dale_out"],
            ),
        ]

        sign_dicts = [
            dict(Wmin=0.0, Wmax=100.0),
            dict(Wmin=-100.0, Wmax=0.0),
        ]

        pop_pre_arr = np.array([], dtype=int)
        pop_post_arr = np.array([], dtype=int)
        weights_arr = np.array([])

        for weight_dict in weight_dicts:
            if weight_dict["constrain_sign"] or weight_dict["constrain_dale"]:
                conns = nest.GetConnections(weight_dict["nrns_pre"], weight_dict["nrns_post"])
                conns_dict = conns.get()
                conns_dict["source"] = np.array(conns_dict["source"], dtype=int)
                conns_dict["target"] = np.array(conns_dict["target"], dtype=int)
                conns_dict["weight"] = np.array(conns_dict["weight"])

                if weight_dict["constrain_dale"]:
                    source_unique = np.unique(conns_dict["source"])
                    proportion_inh = 1.0 / (1.0 + self.config["exc_to_inh_ratio"])
                    sources_inh = np.random.choice(
                        source_unique, int(len(source_unique) * proportion_inh), replace=False
                    )
                    new_weights = []
                    for source, weight in zip(conns_dict["source"], np.abs(conns_dict["weight"])):
                        if source in sources_inh:
                            weight *= -1.0
                        new_weights.append(weight)
                    conns_dict["weight"] = np.array(new_weights)

                nest.Disconnect(conns)

                weights_arr = np.append(weights_arr, conns_dict["weight"])
                pop_pre_arr = np.append(pop_pre_arr, conns_dict["source"])
                pop_post_arr = np.append(pop_post_arr, conns_dict["target"])

        for sign_dict in sign_dicts:
            sign = np.sign(sign_dict["Wmin"])
            label = "positive" if sign >= 0.0 else "negative"

            params_common = params_common_syn_eprop.copy()
            params_common["optimizer"].update(sign_dict)
            params_common["weight"] = sign_dict["Wmin"]

            base_synapse_model = params_syn_base["synapse_model"]
            synapse_model = f"{base_synapse_model}_{label}"
            nest.CopyModel(base_synapse_model, synapse_model, params_common)

            if len(weights_arr) > 0:
                idc = np.where(weights_arr >= 0.0 if sign >= 0.0 else weights_arr < 0.0)

                params_base = params_syn_base.copy()
                params_base["synapse_model"] = synapse_model
                params_base["weight"] = weights_arr[idc]
                params_base["delay"] = np.ones_like(params_base["weight"]) * params_syn_base["delay"]

                nest.Connect(pop_pre_arr[idc], pop_post_arr[idc], conn_spec="one_to_one", syn_spec=params_base)

    def set_synapse_defaults(self, eta):
        for synapse_model in nest.synapse_models:
            if synapse_model.startswith("eprop_synapse"):
                nest.SetDefaults(synapse_model, dict(optimizer=dict(eta=eta)))

    def save_weights_snapshots(self, weights_dict_pre, weights_dict_post):
        for phase, weights_dict in zip(["pre", "post"], [weights_dict_pre, weights_dict_post]):
            for k, v in weights_dict.items():
                label = f"{phase}_train_{k}"
                v_ = {k: v[k] for k in v.keys() if k != "weight_matrix"}
                pd.DataFrame.from_dict(v_).to_csv(self.recordings_dir / f"weights_{label}.csv", index=False)

    def save_weight_recordings(self, sender_nrns, target_nrns, df, label, recorder_label):
        sender_min = min(sender_nrns.tolist())
        sender_max = max(sender_nrns.tolist())
        target_min = min(target_nrns.tolist())
        target_max = max(target_nrns.tolist())

        condition1 = (df.sender >= sender_min) & (df.sender <= sender_max)
        condition2 = (df.targets >= target_min) & (df.targets <= target_max)

        df_sub = df[condition1 & condition2]

        if label != "":
            label = f"_{label}"
        df_sub.to_csv(self.recordings_dir / f"{recorder_label}{label}.csv", index=False)

    def process_recordings(self, duration, nrns_in, nrns_rec, nrns_out):
        recorder_labels = ["multimeter_out"]
        if self.config["record_dynamics"]:
            recorder_labels += ["spike_recorder_in", "weight_recorder"]
            if "evidence" in self.file_name:
                nrn_types = ["reg", "ad"]
            else:
                nrn_types = ["rec"]
            for nrn_type in nrn_types:
                recorder_labels.extend([f"multimeter_{nrn_type}", f"spike_recorder_{nrn_type}"])

        for recorder_label in recorder_labels:
            file_names = sorted(self.recordings_dir.glob(f"{recorder_label}*.dat"))

            dfs = []

            df_file = self.recordings_dir / f"{recorder_label}.csv"
            if df_file.is_file():
                dfs.append(pd.read_csv(df_file, engine="c"))

            for fname in file_names:
                df_new = pd.read_csv(fname, skiprows=2, sep="\t", engine="c")
                if not df_new.empty:
                    dfs.append(df_new)
                    # fname.unlink()

            if not dfs:
                df = df_new
            elif len(dfs) == 1 or (len(dfs) == 2 and dfs[0].empty):
                df = dfs[-1]
            else:
                df = pd.concat(dfs, ignore_index=True)

            if recorder_label == "weight_recorder":
                self.save_weight_recordings(nrns_in + nrns_rec, nrns_rec + nrns_out, df, "", recorder_label)
                self.save_weight_recordings(nrns_in, nrns_rec, df, "in", recorder_label)
                self.save_weight_recordings(nrns_rec, nrns_rec, df, "rec", recorder_label)
                self.save_weight_recordings(nrns_rec, nrns_out, df, "out", recorder_label)

            else:
                df.to_csv(df_file, index=False)

                if "task" in duration.keys():
                    t_margin = 50.0  # ms, record a bit into the next / previous iteration

                    condition_first_iteration = df.time_ms < duration["sequence"] + t_margin

                    condition_last_iteration = df.time_ms >= duration["task"] - duration["sequence"] - t_margin

                    df_subset = df[condition_first_iteration | condition_last_iteration]
                    df_subset.to_csv(self.recordings_dir / f"{recorder_label}_subset.csv", index=False)

    def get_events(self, prefix, save=False):
        data_list = []
        for f in sorted(self.recordings_dir.glob(f"*{prefix}*.dat")):
            data = pd.read_csv(f, delimiter="\t", comment="#", engine="c")
            if not data.empty:
                data_list.append(data)
        events_mm_out = pd.concat(data_list, ignore_index=True)
        if save:
            events_mm_out.to_csv(self.recordings_dir / "multimeter_out.csv", index=False)
        return events_mm_out

    def get_results(self):
        return pd.read_csv(self.recordings_dir / "learning_performance.csv", engine="c")

    def save_timing(self, kernel_status):
        for k in self.timing_dict.keys():
            v = kernel_status[k]
            if k == "time_simulate":
                self.timing_dict[k] += v
            else:
                self.timing_dict[k] = v

        timing_dict_final = {}
        for k, v in self.timing_dict.items():
            if k == "biological_time":
                v /= 1000.0  # convert from ms to s
            timing_dict_final[f"{k}_s"] = [v]

        pd.DataFrame.from_dict(timing_dict_final).to_csv(self.recordings_dir / "timing.csv", index=False)
    def save_phase(self, phase_label, n_iter):
        with open(self.recordings_dir / "phases.dat", "a") as f:
            f.write(f"{phase_label},{n_iter}\n")

    def save_performance(self, save, loss, errors):
        if save:
            df = pd.read_csv(self.recordings_dir / "phases.dat", names=["phase", "repetition"], engine="c")
            labels = np.repeat(df.phase.values, df.repetition.values).tolist()
            pd.DataFrame(dict(
                iteration=np.arange(len(loss)),
                loss=loss,
                error=errors,
                label=labels,
                )
            ).to_csv(self.recordings_dir / "learning_performance.csv", index=False)

    def verify(self):
        # print(self.file_name)
        # for l in self.loss:
        #     print(f"{l:.14f},")
        # exit()
        loss = np.array(self.loss)

        if self.file_name == "eprop_supervised_classification_evidence-accumulation_bsshslm_2020.py":
            loss_reference = [
                0.74115255000619,
                0.74038818770074,
                0.66578523317777,
                0.66364419332299,
                0.72942896284495,
                0.65825443888416,
            ]
        elif self.file_name == "eprop_supervised_classification_evidence-accumulation.py":
            loss_reference = [
                34.58427289782617,
                36.87835068653019,
                28.89970643558962,
                31.60581680525203,
                36.76571948680768,
                29.90618754038629,
            ]
        elif self.file_name == "eprop_supervised_regression_sine-waves_bsshslm_2020.py":
            loss_reference = [
                101.96435699904158,
                103.46673112620579,
                103.34060707477168,
                103.68024403768638,
                104.41277574875247,
            ]
        elif self.file_name == "eprop_supervised_regression_sine-waves.py":
            loss_reference = [
                107.73732072362752,
                106.42253313316886,
                107.37869441301808,
                108.10839027499375,
                107.76400611943626,
            ]
        elif self.file_name == "eprop_supervised_classification_neuromorphic_mnist.py":
            loss_reference = [
                0.49569090581695,
                0.52751321436889,
                0.51467659566501,
                0.50595422166446,
                0.50532549825770,
                0.49938869752847,
            ]
        elif self.file_name == "eprop_supervised_classification_neuromorphic_mnist_bsshslm_2020.py":
            loss_reference = [
                2.29926915739071,
                2.30735389920452,
                2.31229167547814,
                2.30398946726470,
                2.30571008112245,
                2.30277356036807,
            ]
        elif self.file_name == "eprop_supervised_regression_lemniscate_bsshslm_2020.py":
            loss_reference = [
                314.30442538643001,
                313.84127193622919,
                312.33971633807948,
                310.66410755892281,
                309.19353500432857,
            ]
        elif self.file_name == "eprop_supervised_regression_handwriting_bsshslm_2020.py":
            loss_reference = [
                91.40191610510351,
                90.53583357361666,
                89.91415022333089,
                88.54544175584950,
                86.98770239575573,
            ]

        n_compare = min(len(loss), len(loss_reference))
        verification_successful = np.allclose(loss[:n_compare], loss_reference[:n_compare], atol=1e-14, rtol=0)

        if not verification_successful:
            deviation_idc = np.where(loss[:n_compare] != loss_reference[:n_compare])[0]
            for deviation_idx in deviation_idc:
                print(f"{deviation_idx}. iteration")
                print(f"{loss[deviation_idx]:.16f} loss")
                print(f"{loss_reference[deviation_idx]:.16f} reference loss")
                print(f"{loss[deviation_idx]-loss_reference[deviation_idx]:.16f} delta")
        print(verification_successful)
